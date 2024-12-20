use super::Slab;
use core::sync::atomic;

type BitmapElement = atomic::AtomicU64;

macro_rules! bmap_elem_bits {
    () => {
        (::core::mem::size_of::<$crate::slab_meta::BitmapElement>() * 8)
    };
}

/// Returns the number ob [BitmapElement]'s in the metadata bitmap.
pub const fn meta_bitmap_size<T>(slab_size: usize) -> usize {
    assert!(slab_size >= size_of::<T>() * 2);
    (slab_size / size_of::<T>()).div_ceil(size_of::<BitmapElement>() * 8)
}

/// Returns the size of a `SlabMeta<T,slab_size>`.
///
/// This is required because the `generic_const_exprs` feature is buggy, specifically it is
/// required because the compiler cant resolve `size_of<SlabMetadata<T,_>` in while trying to
/// resolve another const bound. This prevents needing to resolve `size_of<SlabMetadata<T,_>`.
pub(crate) const fn size_of_meta<T>(slab_size: usize) -> usize {
    const META_ALIGN: usize = 8;
    let size = (META_ALIGN * 2) + (meta_bitmap_size::<T>(slab_size) * size_of::<BitmapElement>());

    let t = if size & META_ALIGN - 1 != 0 {
        (size & !(META_ALIGN - 1)) + META_ALIGN
    } else {
        size
    };

    t
}

#[repr(C)]
pub(crate) struct SlabMetadata<T, const SLAB_SIZE: usize>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); super::slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    next: atomic::AtomicPtr<Slab<T, SLAB_SIZE>>,
    prev: atomic::AtomicPtr<Slab<T, SLAB_SIZE>>,

    // Array element type here is a trade between time and space complexity.
    // Larger types have higher space and lower time complexity
    // The first bit in the first element is a lock bit. This bit would otherwise indicate the element `self` is stored in
    // For `1..Self::meta_size_elements()` bits should be set to `1`
    bitmap: [BitmapElement; meta_bitmap_size::<T>(SLAB_SIZE)],
}

impl<T, const SLAB_SIZE: usize> SlabMetadata<T, SLAB_SIZE>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); super::slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    pub(crate) fn new() -> Self
    where
        [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
        [(); super::slab_count_obj_elements::<T, SLAB_SIZE>()]:,
    {
        let r = Self {
            next: atomic::AtomicPtr::new(core::ptr::null_mut()),
            prev: atomic::AtomicPtr::new(core::ptr::null_mut()),

            bitmap: [const { BitmapElement::new(u64::MAX) }; meta_bitmap_size::<T>(SLAB_SIZE)],
        };

        // todo optimize this
        for i in Self::reserved_bits()
            ..super::slab_count_obj_elements::<T, SLAB_SIZE>() + Self::reserved_bits()
        {
            r.free(i)
        }

        r
    }

    /// Determines if [Self::alloc] will fail.
    /// the result of [Self::alloc] is not guaranteed unless the caller has exclusive access to `self`.
    ///
    /// ## Concurrency
    ///
    /// This function has no specific ordering.
    ///
    /// `self` may be modified to invalidate the result before this function exists.
    /// Thus, the result should be treated as a best guess.
    pub fn is_full(&self) -> bool {
        for i in &self.bitmap {
            if i.load(atomic::Ordering::Relaxed) < u64::MAX {
                return false;
            }
        }
        true
    }

    pub fn count_free(&self) -> usize {
        let mut count = 0;
        for i in &self.bitmap {
            count += i.load(atomic::Ordering::Relaxed).count_zeros();
        }
        count as usize
    }

    /// Returns the number of elements required to store `self` within a slab.
    ///
    /// When accessing the bitmap in `self` `Self::meta_size_elements()` should be ignored.
    ///
    /// This will never return `0`
    pub const fn reserved_bits() -> usize {
        size_of::<Self>().div_ceil(calc_obj_size::<T>())
    }

    /// Attempts to Allocate an object, returns its slab index.
    pub fn alloc(&self) -> Option<usize> {
        // Skips elements which should not be checked because they are never free.as
        //
        let first_element = Self::reserved_bits() / bmap_elem_bits!();

        'map: for i in first_element..self.bitmap.len() {
            // nested loop, 'element will retry this element until it either it locks a bit or exhausts the element
            'element: loop {
                let t = self.bitmap[i].load(atomic::Ordering::Relaxed);

                let first_zero = t.trailing_ones() as usize;
                if first_zero == element_bits() {
                    continue 'map; // next element
                } else {
                    let mask = 1 << first_zero;
                    // acquire the bit
                    let ret = self.bitmap[i].fetch_or(mask, atomic::Ordering::Acquire);
                    if ret & mask == 0 {
                        // check the *we* acquired the bit
                        return Some(first_zero + (i * bmap_elem_bits!()));
                    }
                    // we didn't acquire the bit, try a new one
                    continue 'element;
                }
            }
        }
        None
    }

    /// Sets the value of the requested bit to `value`
    pub(crate) fn set_bit(&self, index: usize, value: bool) {
        let byte = index / (size_of::<BitmapElement>() * 8);
        let bit = 1 << index % (size_of::<BitmapElement>() * 8);

        if value {
            self.bitmap[byte].fetch_or(bit, atomic::Ordering::Relaxed);
        } else {
            self.bitmap[byte].fetch_and(!bit, atomic::Ordering::Relaxed);
        }
    }

    /// Returns the value of the requested bit
    fn get_bit(&self, index: usize) -> bool {
        let byte = index / (size_of::<BitmapElement>() * 8);
        let bit = 1 << index % (size_of::<BitmapElement>() * 8);

        self.bitmap[byte].load(atomic::Ordering::Relaxed) & bit != 0
    }

    pub(crate) fn prev_slab(&self) -> Option<*mut Slab<T, SLAB_SIZE>> {
        let t = self.prev.load(atomic::Ordering::Relaxed);
        if t.is_null() {
            None
        } else {
            Some(t)
        }
    }

    pub(crate) fn next_slab(&self) -> Option<*mut Slab<T, SLAB_SIZE>> {
        let t = self.next.load(atomic::Ordering::Relaxed);
        if t.is_null() {
            None
        } else {
            Some(t)
        }
    }

    /// Replaces the pointer to the previous slab with `next`. Returns the previous value.
    pub(crate) fn set_prev(
        &self,
        prev: Option<*mut Slab<T, SLAB_SIZE>>,
    ) -> Option<*mut Slab<T, SLAB_SIZE>> {
        let ret = if let Some(p) = prev {
            self.prev.swap(p, atomic::Ordering::Relaxed)
        } else {
            self.prev
                .swap(core::ptr::null_mut(), atomic::Ordering::Relaxed)
        };
        if ret.is_null() {
            None
        } else {
            Some(ret)
        }
    }

    /// Replaces the pointer to the next slab with `next`. Returns the previous value.
    pub(crate) fn set_next(
        &self,
        next: Option<*mut Slab<T, SLAB_SIZE>>,
    ) -> Option<*mut Slab<T, SLAB_SIZE>> {
        let ret = if let Some(p) = next {
            self.next.swap(p, atomic::Ordering::Relaxed)
        } else {
            self.next
                .swap(core::ptr::null_mut(), atomic::Ordering::Relaxed)
        };

        if ret.is_null() {
            None
        } else {
            Some(ret)
        }
    }

    pub(crate) fn free(&self, index: usize) {
        self.set_bit(index, false);
    }

    #[cfg(debug_assertions)]
    #[allow(dead_code)]
    #[doc(hidden)]
    pub fn bitmap(&self) -> &[BitmapElement] {
        &self.bitmap
    }
}

const fn calc_obj_size<T>() -> usize {
    let size = size_of::<T>();
    let align = align_of::<T>();

    if size & (align - 1) != 0 {
        (size & (!align - 1)) + align
    } else {
        size
    }
}

/// Returns the number of bits in [BitmapElement]
const fn element_bits() -> usize {
    size_of::<BitmapElement>() * 8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_obj_size() {
        assert_eq!(calc_obj_size::<u8>(), 1);
        assert_eq!(calc_obj_size::<u16>(), 2);
        assert_eq!(calc_obj_size::<u32>(), 4);
        assert_eq!(calc_obj_size::<u64>(), 8);

        #[repr(align(64))]
        struct TestAlign {
            _i: u8,
        }

        assert_eq!(calc_obj_size::<TestAlign>(), 64);

        struct TestAlign2 {
            _i: u64,
            _u: u32,
        }

        assert_eq!(calc_obj_size::<TestAlign2>(), 16);

        #[repr(align(2))]
        struct TestAlign3;

        assert_eq!(calc_obj_size::<TestAlign3>(), 0)
    }

    #[test]
    fn check_reserved_bits() {
        let m = SlabMetadata::<u16, 4096>::new();
        for i in 0..2048 {
            assert_eq!(
                m.get_bit(i),
                i < SlabMetadata::<u16, 4096>::reserved_bits(),
                "Iteration {i}, res: {}",
                SlabMetadata::<u16, 4096>::reserved_bits()
            );
        }
        let m = SlabMetadata::<u16, 64>::new();
        for i in 0..4 {
            assert_eq!(
                m.get_bit(i),
                i < SlabMetadata::<u16, 64>::reserved_bits(),
                "Iteration {i}, res: {}",
                SlabMetadata::<u16, 4096>::reserved_bits()
            );
        }
        let m = SlabMetadata::<u128, 64>::new();
        for i in 0..4 {
            assert_eq!(
                m.get_bit(i),
                i < SlabMetadata::<u128, 64>::reserved_bits(),
                "Iteration {i}, res: {}",
                SlabMetadata::<u16, 4096>::reserved_bits()
            );
        }
    }

    /// We don't really need this, it's just here to assert that the required components of
    /// `generic_const_exprs` are working correctly.
    #[test]
    fn correct_const_size_of() {
        struct TestStruct<T>
        where
            [(); size_of::<T>()]:,
        {
            _p: [bool; size_of::<T>()],
            _phantom: core::marker::PhantomData<T>,
        }

        fn new<T>() -> TestStruct<T>
        where
            [(); size_of::<T>()]:,
        {
            TestStruct {
                _p: [false; size_of::<T>()],
                _phantom: Default::default(),
            }
        }

        assert_eq!(new::<u8>()._p.len(), 1);
        assert_eq!(new::<[u8; 6]>()._p.len(), 6);
        assert_eq!(new::<std::ptr::NonNull<usize>>()._p.len(), 8);
    }

    #[test]
    fn test_meta_bitmap_size() {
        assert_eq!(meta_bitmap_size::<u8>(2), 1);
        assert_eq!(meta_bitmap_size::<u8>(8), 1);
        assert_eq!(meta_bitmap_size::<u8>(16), 1);
        assert_eq!(meta_bitmap_size::<u8>(32), 1);
        assert_eq!(meta_bitmap_size::<u8>(128), 2);
        assert_eq!(meta_bitmap_size::<u8>(256), 4);
        assert_eq!(meta_bitmap_size::<u64>(16), 1);
    }

    #[test]
    #[should_panic]
    /// Slab must contain at least 2 object-elements.
    /// <u8,1> only contains one element
    fn test_bad_meta_value() {
        meta_bitmap_size::<u8>(1);
    }

    #[test]
    fn test_metadata_geom() {
        let meta = SlabMetadata::<u128, 4096>::new();
        assert_eq!(meta.bitmap.len() * size_of::<BitmapElement>() * 8, 256); // `len * size_of * 8` gets number of obj_elements
    }

    #[test]
    fn test_manual_sizing() {
        macro_rules! assert_meta_size {
            ($ty:path,$ss:literal) => {
                assert_eq!(
                    size_of_meta::<$ty>($ss),
                    size_of::<SlabMetadata<$ty, $ss>>()
                );
            };
        }

        //assert_meta_size!(u8, 32);
        assert_meta_size!(u8, 64);
        assert_meta_size!(u128, 128);
    }
}
