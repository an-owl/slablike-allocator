#![cfg_attr(not(test), no_std)]
#![feature(allocator_api)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

//! The slab-like allocator works using a method inspired by a SLAB allocator.
//! It uses regions of memory of `SLAB_SIZE` bytes divided up into object elements.
//! Each region (slab) is located using a self-contained doubly-linked list. The head contains a
//! pointer to the first and last slab, as a doubly linked list normally would.
//! However, it also maintains a cursor into a slab which will be used to allocate memory from next.
//!
//! Each slab is divided into object-elements (OE), and metadata. The metadata is allocated into the
//! first object elements within the slab. The metadata contains pointers to the previous and next
//! slabs and a free list.
//! The free list is a bitmap of OE, because the metadata struct is allocated to at
//! least one OE the first bit is used as a lock bit. To locate a free OE the leading ones
//! will be counted until a zero is found, which is set to one and the lock bit is cleared.
//!
//! If a slab is full the cursor will be moved to the next slab and which will be searched for a free OE.
//! Using this method full slabs will be moved "before" the cursor and empty slabs are after it reducing seek times.
//! If no free slabs are available then one will be allocated from the given allocator.
//!
//! The list can be sorted, this look back and move all slabs which aren't full ahead of the cursor.
//! Slabs ahead of the cursor are guaranteed to have free space so these do not need to be checked.

mod slab_meta;

use core::alloc::{AllocError, Layout};
use core::ptr::NonNull;
use slab_meta::SlabMetadata;

pub use slab_meta::meta_bitmap_size;

pub struct SlabLike<A: core::alloc::Allocator, T, const SLAB_SIZE: usize>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    alloc: A,

    // When a thread attempts to allocate memory it must increment the visitors counter.
    // If the allocation of the current cursor fails
    visitors: core::sync::atomic::AtomicUsize,

    pointers: spin::RwLock<SlabLikePointers<T, SLAB_SIZE>>,
}

struct SlabLikePointers<T, const SLAB_SIZE: usize>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    // Head and tail must always be the same variant
    head: Option<NonNull<Slab<T, SLAB_SIZE>>>,
    tail: Option<NonNull<Slab<T, SLAB_SIZE>>>,
    cursor: Option<NonNull<Slab<T, SLAB_SIZE>>>,
}

impl<A: core::alloc::Allocator, T: 'static, const SLAB_SIZE: usize> SlabLike<A, T, SLAB_SIZE>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    pub const fn new(alloc: A) -> Self {
        Self {
            alloc,
            visitors: core::sync::atomic::AtomicUsize::new(0),
            pointers: spin::RwLock::new(SlabLikePointers {
                head: None,
                tail: None,
                cursor: None,
            }),
        }
    }

    fn new_slab(&self) -> Result<&'static mut Slab<T, SLAB_SIZE>, AllocError> {
        let ptr = self
            .alloc
            .allocate(Layout::from_size_align(size_of::<Slab<T, SLAB_SIZE>>(), SLAB_SIZE).unwrap());
        let slab_ptr = ptr?.cast();

        unsafe { slab_ptr.write(Slab::<T, SLAB_SIZE>::new()) };
        let n_slab: &mut Slab<T, SLAB_SIZE> = unsafe { &mut *ptr?.as_ptr().cast() };
        Ok(n_slab)
    }

    /// Returns a reference to the last slab owned by self.
    fn tail(&self) -> Option<&mut Slab<T, SLAB_SIZE>> {
        self.pointers
            .read()
            .tail
            .map(|ptr| unsafe { &mut *ptr.as_ptr().cast() })
    }

    /// Returns a reference to the first slab owned by self
    fn head(&self) -> Option<&mut Slab<T, SLAB_SIZE>> {
        self.pointers
            .read()
            .head
            .map(|ptr| unsafe { &mut *ptr.as_ptr().cast() })
    }

    /// Returns a reference to the slab pointed at by the cursor.
    fn cursor(&self) -> Option<&mut Slab<T, SLAB_SIZE>> {
        self.pointers
            .read()
            .head
            .map(|ptr| unsafe { &mut *ptr.as_ptr().cast() })
    }
}

impl<T, const SLAB_SIZE: usize> SlabLikePointers<T, SLAB_SIZE>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    /// Updates the cursor to point to `next`.
    /// This method should be called when `self.cursor` failed to allocate memory and the visitor count is zero.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `next` is contained within `self`'s linked list.
    unsafe fn advance_cursor(&mut self, next: Option<&mut Slab<T, SLAB_SIZE>>) {
        self.cursor = next.map(|p| NonNull::new(p).unwrap());
    }

    /// Locates a slab that the caller will allocate memory from and determines whether the cursor must be updated.
    ///
    /// If no slab can be found then the caller should allocate a new slab and append it to the linked list.
    fn get_usable_slab(&self, visitor: usize) -> Option<(&mut Slab<T, SLAB_SIZE>, bool)> {
        let slab = self.cursor.map(|p| unsafe { &mut *p.as_ptr() })?;

        let slab_r = self.locate_available_slab(visitor, slab)?;

        // returns good slab, checks that
        let dirty = core::ptr::eq(slab_r, self.cursor.unwrap().as_ptr());
        Some((slab_r, dirty))
    }

    /// Attempts to locate a slab which can be used to allocate new data from.
    ///
    /// If no slab can be located, then the last slab will be returned. The caller can append a new
    /// slab to the list and call using the previously returned value
    fn locate_available_slab<'a>(
        &'a self,
        mut visitors: usize,
        mut slab: &'a mut Slab<T, SLAB_SIZE>,
    ) -> Option<&'a mut Slab<T, SLAB_SIZE>> {
        loop {
            // We know how many free OEs are in each slab, so we subtract the visitor count by that until it underflows.
            // This slab is guaranteed to have a free OE.
            match visitors.checked_sub(slab.query_free()) {
                Some(n) => {
                    visitors = n;
                    slab = slab.next_slab()?
                }
                None => return Some(slab),
            }
        }
    }

    /// Allocates a new slab, appends it to the end of the linked list.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `slab` is not already within the list.
    unsafe fn new_append(&mut self, slab: &'static mut Slab<T, SLAB_SIZE>) {
        if self.head.is_none() {
            // Only slab, initialize all pointers
            self.head = Some(NonNull::new(slab as *mut Slab<T, SLAB_SIZE>).unwrap());
            self.tail = Some(NonNull::new(slab as *mut Slab<T, SLAB_SIZE>).unwrap());
            self.cursor = Some(NonNull::new(slab as *mut Slab<T, SLAB_SIZE>).unwrap());
        } else {
            let tail = unsafe { &mut *self.tail.expect("Tail not initialized").as_ptr() };
            let _r = tail.set_next(Some(slab));
            debug_assert!(_r.is_none());
            self.tail = Some(NonNull::new(slab as *mut Slab<T, SLAB_SIZE>).unwrap());
        }
    }
}

unsafe impl<A: core::alloc::Allocator, T: 'static, const SLAB_SIZE: usize> core::alloc::Allocator
    for SlabLike<A, T, SLAB_SIZE>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert_eq!(
            layout,
            Layout::new::<T>(),
            "Unexpected layout for {}: {layout:?}, expected {:?}",
            core::any::type_name::<Self>(),
            Layout::new::<T>()
        );

        let visitors = self
            .visitors
            .fetch_add(1, core::sync::atomic::Ordering::Acquire);

        let mut rl = self.pointers.upgradeable_read();
        let (ptr, slab, dirty) = match rl.get_usable_slab(visitors) {
            Some((slab, dirty)) => (slab.alloc()?, slab, dirty),
            None => {
                let ns = self.new_slab()?;
                let rc = ns.alloc()?;
                // SAFETY: This is guaranteed to not be within the existing list.
                let mut wl = rl.upgrade();
                unsafe { wl.new_append(ns) };
                let ret = (rc, unsafe { &mut *wl.tail.unwrap().as_ptr() }, true);
                rl = wl.downgrade_to_upgradeable();
                ret
            }
        };

        // SAFETY: This disassociates `slab`'s lifetime from `rl`.
        // rl.upgrade() consumes `rl` but `slab` is bounded to it.
        // `slab` is taken as mutable but is not mutated after `rl` is dropped.
        let slab = unsafe {
            core::mem::transmute::<&mut Slab<T, SLAB_SIZE>, &'static mut Slab<T, SLAB_SIZE>>(slab)
        };

        if self
            .visitors
            .fetch_sub(1, core::sync::atomic::Ordering::Release)
            == 1
            && dirty
        {
            unsafe { rl.upgrade().advance_cursor(Some(slab)) }
        };

        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        debug_assert_eq!(
            layout,
            Layout::new::<T>(),
            "Unexpected layout for {}: {layout:?}, expected {:?}",
            core::any::type_name::<Self>(),
            Layout::new::<T>()
        );

        // This takes `ptr` aligns it down to `SLAB_SIZE` and casts it to Slab
        // Because `Self` forces alignment of Slabs to `SLAB_SIZE` we can guarantee that this is indeed the pointer into a Slab.
        let tgt_slab = unsafe {
            &mut *(((ptr.as_ptr() as usize) & !(SLAB_SIZE - 1)) as *mut Slab<T, SLAB_SIZE>)
        };

        // offset into array
        let offset = (ptr.as_ptr() as usize) - ((&tgt_slab.obj_elements) as *const _ as usize);
        let index = if offset == 0 {
            0
        } else {
            // offset divided by element size, returns element index
            offset / (tgt_slab.obj_elements.len()) / size_of_val(&tgt_slab.obj_elements)
        };

        tgt_slab.free(index)
    }
}

/// Returns the number of object elements that can be stored in a [Slab]
///
/// This fn will panic if the number of object elements is 0.
pub const fn slab_count_obj_elements<T, const SLAB_SIZE: usize>() -> usize {
    let elem_size = size_of::<[T; 2]>() / 2;
    let raw_obj_elem_count = SLAB_SIZE / elem_size;

    let meta_size = slab_meta::size_of_meta::<T>(SLAB_SIZE);
    let meta_obj_elem_size = meta_size.div_ceil(elem_size);

    let num_obj_elements = raw_obj_elem_count - meta_obj_elem_size;
    assert!(
        num_obj_elements > 0,
        "Slab must have more than one Object elements"
    ); // This'd be a shit allocator if we had 100% overhead

    num_obj_elements
}

#[repr(C)]
struct Slab<T, const SLAB_SIZE: usize>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    slab_metadata: SlabMetadata<T, SLAB_SIZE>,
    obj_elements: [core::mem::MaybeUninit<T>; slab_count_obj_elements::<T, SLAB_SIZE>()],
}

impl<T, const SLAB_SIZE: usize> Slab<T, SLAB_SIZE>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    pub fn new() -> Self
    where
        [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
        [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
    {
        Self {
            slab_metadata: SlabMetadata::new(),
            obj_elements: [const { core::mem::MaybeUninit::uninit() };
                slab_count_obj_elements::<T, SLAB_SIZE>()],
        }
    }

    fn alloc(&mut self) -> Result<NonNull<[u8]>, AllocError> {
        let index = self.slab_metadata.alloc().ok_or(AllocError)?
            - SlabMetadata::<T, SLAB_SIZE>::reserved_bits();

        // uses get() because the index may be out of bounds. We check this here because im sure the
        // compiler will bitch if SlabMetadata does it.
        let u =
            self.obj_elements.get_mut(index).ok_or(AllocError)? as *mut core::mem::MaybeUninit<T>;
        // SAFETY: Pointer is taken from a reference
        Ok(unsafe {
            NonNull::new_unchecked(core::slice::from_raw_parts_mut(u.cast(), size_of::<T>()))
        })
    }

    fn free(&self, index: usize) {
        let bit = slab_meta::SlabMetadata::<T, SLAB_SIZE>::reserved_bits() + index;
        self.slab_metadata.free(bit);
    }

    fn next_slab<'a, 'b>(&'a self) -> Option<&'b mut Self> {
        Some(unsafe { &mut *self.slab_metadata.next_slab()? })
    }

    fn prev_slab(&self) -> Option<&mut Self> {
        Some(unsafe { &mut *self.slab_metadata.prev_slab()? })
    }

    fn set_next(&self, slab: Option<&mut Self>) -> Option<&mut Self> {
        self.slab_metadata
            .set_next(slab.map(|p| p as *mut _))
            .map(|p| unsafe { &mut *p })
    }
    fn set_prev(&self, slab: Option<&mut Self>) -> Option<&mut Self> {
        self.slab_metadata
            .set_prev(slab.map(|p| p as *mut _))
            .map(|p| unsafe { &mut *p })
    }

    fn insert_next(&mut self, new: Option<&mut Self>) {
        new.as_ref().unwrap().set_prev(Some(self));
        if let Some(old_next) = self.set_next(new) {}
    }

    /// Returns the number of free object elements in `self`
    fn query_free(&self) -> usize {
        slab_count_obj_elements::<T, SLAB_SIZE>() - self.slab_metadata.allocated()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_slab_obj_count() {
        let slab = Slab::<u16, 64>::new();
        //assert_eq!(core::mem::size_of::<Slab<u128, 64>>(), 64);
        assert_eq!(slab.obj_elements.len(), 16);
        let slab = Slab::<u128, 64>::new();
        assert_eq!(core::mem::size_of::<Slab<u128, 64>>(), 64);
        assert_eq!(slab.obj_elements.len(), 2);
    }

    #[test]
    fn check_slab_geom_size() {
        assert_eq!(size_of::<Slab<u8, 64>>(), 64);
        assert_eq!(size_of::<Slab<u8, 128>>(), 128);
        assert_eq!(size_of::<Slab<u128, 128>>(), 128);
        assert_eq!(size_of::<Slab<u64, 4096>>(), 4096);
    }

    #[test]
    fn obj_elem_count() {
        assert_eq!(slab_count_obj_elements::<u128, 64>(), 2);
        assert_eq!(slab_count_obj_elements::<u16, 64>(), 16);
    }

    #[test]
    fn test_slab_alloc() {
        let mut slab = Slab::<u128, 64>::new();
        let slab_addr = &slab as *const _ as usize;
        let first = slab.alloc().unwrap();
        assert_eq!(
            first.cast::<u8>().as_ptr() as usize,
            slab_addr + 32,
            "base: {slab_addr:x}"
        );
        let second = slab.alloc().unwrap();
        assert_eq!(
            second.cast::<u8>().as_ptr() as usize,
            slab_addr + 48,
            "base: {slab_addr:x}"
        );
    }

    #[test]
    fn test_slab_alloc_u16() {
        let mut slab = Slab::<u16, 4096>::new();
        let slab_addr = &slab as *const _ as usize;
        let base = &slab as *const _ as usize;
        for i in 0..1776 {
            // 1776 is expected number of obj-elements
            let ptr = slab.alloc().unwrap().cast::<u8>().as_ptr() as usize;
            assert_eq!(
                ptr,
                base + 280 + (2 * i),
                "Iteration {i}, base {base:#x}, meta_end {:#x}",
                base + 272
            );
        }
    }

    #[test]
    fn init_slablike() {
        use std::alloc::Global;
        let sl = SlabLike::<Global, u16, 4096>::new(Global);

        let _b = Box::try_new_in(0u16, &sl).unwrap();

        let sl = SlabLike::<Global, u128, 64>::new(Global);

        let _b = Box::try_new_in(0u128, &sl).unwrap();
    }

    #[test]
    fn slablike_free() {
        use std::alloc::Global;
        let sl = SlabLike::<Global, u16, 4096>::new(Global);

        let b = Box::try_new_in(0u16, &sl).unwrap();
        let addr = &*b as *const _ as usize;
        drop(b);
        let b = Box::try_new_in(0u16, &sl).unwrap();
        assert_eq!(&*b as *const _ as usize, addr);
    }

    #[test]
    fn slablike_auto_extend() {
        use std::alloc::Global;
        let sl = SlabLike::<Global, u128, 64>::new(Global);

        for i in 0..8 {
            let b = Box::try_new_in(0u128, &sl).expect(&std::format!("Failed on iteration {i}"));
            let _ = Box::leak(b);
        }
    }
}
