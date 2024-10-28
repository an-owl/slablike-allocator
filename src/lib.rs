#![cfg_attr(not(test),no_std)]
#![feature(allocator_api)]
#![feature(generic_const_exprs)]

use core::ptr::NonNull;
use core::sync::atomic;

type BitmapElement = atomic::AtomicU8;

pub struct SlabLike<A: core::alloc::Allocator, T, const SLAB_SIZE: usize> {
    alloc: A,
    head: Option<NonNull<Slab<T, SLAB_SIZE>>>,
}

#[repr(C)]
struct Slab<T, const SLAB_SIZE: usize> {
    slab_metadata: SlabMetadata<T, SLAB_SIZE>,


}

#[repr(packed)]
struct SlabMetadata<T, const SLAB_SIZE: usize> where [(); size_of::<T>()/SLAB_SIZE]:  {
    next: Option<NonNull<Slab<T, SLAB_SIZE>>>,
    prev: Option<NonNull<Slab<T, SLAB_SIZE>>>,

    // Array element type here is a trade between time and space complexity.
    // Larger types have higher space and lower time complexity
    // The first bit in the first element is a lock bit. This bit would otherwise indicate the element `self` is stored in
    // For `1..Self::meta_size_elements()` bits should be set to `1`
    bitmap: [BitmapElement; (size_of::<T>()/SLAB_SIZE) / size_of::<BitmapElement>()],
}

impl<T,const SLAB_SIZE: usize> SlabMetadata<T,SLAB_SIZE> {
    const fn new() -> Self {
        let r = Self {
            next: None,
            prev: None,

            bitmap: [BitmapElement::new(0); SLAB_SIZE / (size_of::<BitmapElement>() * 8)],
        };

        // todo optimize this
        for i in Self::reserved_bits() {
            r.set_bit(i, true);
        }

        r
    }

    /// Returns the number of elements required to store `self` within a slab.
    ///
    /// When accessing the bitmap in `self` `Self::meta_size_elements()` should be ignored.
    ///
    /// This will never return `0`
    const fn reserved_bits() -> usize {
        size_of::<Self>().div_ceil(calc_obj_size::<T>())
    }

    /// Attempts to Allocate an object, returns its slab index.
    fn alloc(&self) -> Option<usize> {
        while self.bitmap[0].fetch_or(1, atomic::Ordering::Acquire) | 1 == 0 {
            core::hint::spin_loop();
        }

        // Skips elements which should not be checked because they are never free.as
        //
        let first_element = Self::reserved_bits() / size_of::<BitmapElement>();

        let mut obj_elem = None;

        for i in first_element..self.bitmap.len() {
            let t = self.bitmap[i].load(atomic::Ordering::Relaxed);

            let first_zero = t.leading_ones() as usize;
            if first_zero == element_bits() {
                continue;
            } else {
                obj_elem = Some(i);
            }
        }
        obj_elem
    }

    /// Sets the value of the requested bit to `value`
    fn set_bit(&self, index: usize, value: bool) {
        let byte = index / size_of::<BitmapElement>() * 8;
        let bit = 1 << index % size_of::<BitmapElement>() * 8;

        if value {
            self.bitmap[byte].fetch_or(bit, atomic::Ordering::Relaxed);
        } else {
            self.bitmap[byte].fetch_and(!bit, atomic::Ordering::Relaxed);
        }
    }

    /// Returns the value of the requested bit
    fn get_bit(&self, index: usize) -> bool {
        let byte = index / size_of::<BitmapElement>() * 8;
        let bit = 1 << index % size_of::<BitmapElement>() * 8;

        self.bitmap[byte].load(atomic::Ordering::Relaxed) & bit != 0
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
            _i: u8
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

    // We don't really need this, it's just here to assert that the required components of
    // `generic_const_exprs` is working correctly.
    #[test]
    fn correct_const_size_of() {
        struct TestStruct<T> where [(); size_of::<T>()]: {
            _p: [bool; size_of::<T>()],
            _phantom: core::marker::PhantomData<T>,
        }

        fn new<T>() -> TestStruct<T> where [(); size_of::<T>()]: {
            TestStruct {
                _p: [false; size_of::<T>()],
                _phantom: Default::default(),
            }
        }

        assert_eq!(new::<u8>()._p.len(), 1);
        assert_eq!(new::<[u8;6]>()._p.len(), 6);
        assert_eq!(new::<NonNull<usize>>()._p.len(), 8);
    }

}