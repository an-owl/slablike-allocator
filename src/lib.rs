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
    lock: core::sync::atomic::AtomicBool,

    // Head and tail must always be the same variant
    head: core::sync::atomic::AtomicPtr<Slab<T, SLAB_SIZE>>,
    tail: core::sync::atomic::AtomicPtr<Slab<T, SLAB_SIZE>>,
    cursor: core::sync::atomic::AtomicPtr<Slab<T, SLAB_SIZE>>,
}

impl<A: core::alloc::Allocator, T, const SLAB_SIZE: usize> SlabLike<A, T, SLAB_SIZE>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    pub const fn new(alloc: A) -> Self {
        Self {
            alloc,
            lock: core::sync::atomic::AtomicBool::new(false),
            visitors: core::sync::atomic::AtomicUsize::new(0),
            head: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
            tail: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
            cursor: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
        }
    }

    /// Allocates a new slab, appends it to the end of the linked list
    fn new_slab(&self) -> Result<(), AllocError> {
        let ptr = self
            .alloc
            .allocate(Layout::from_size_align(size_of::<Slab<T, SLAB_SIZE>>(), SLAB_SIZE).unwrap());
        let n_slab: &mut Slab<T, SLAB_SIZE> = unsafe { &mut *ptr?.as_ptr().cast() };

        while self
            .lock
            .compare_exchange_weak(
                false,
                true,
                core::sync::atomic::Ordering::Acquire,
                core::sync::atomic::Ordering::Relaxed,
            )
            .is_err()
        {
            core::hint::spin_loop()
        }

        // Only slab, initialize all pointers
        if self
            .head
            .load(core::sync::atomic::Ordering::Relaxed)
            .is_null()
        {
            self.head.store(
                (n_slab as *mut Slab<T, SLAB_SIZE>).cast(),
                core::sync::atomic::Ordering::Relaxed,
            );
            self.tail.store(
                (n_slab as *mut Slab<T, SLAB_SIZE>).cast(),
                core::sync::atomic::Ordering::Relaxed,
            );
            self.cursor.store(
                (n_slab as *mut Slab<T, SLAB_SIZE>).cast(),
                core::sync::atomic::Ordering::Relaxed,
            );
        } else {
            let tail = self.tail().expect("Tail not initialized");
            debug_assert!(tail.set_next(Some(n_slab)).is_none());
            tail.set_next(Some(n_slab));
            self.tail
                .store(n_slab, core::sync::atomic::Ordering::Relaxed);
        }

        self.lock
            .store(false, core::sync::atomic::Ordering::Release);

        Ok(())
    }

    /// Returns a reference to the last slab owned by self.
    fn tail(&self) -> Option<&mut Slab<T, SLAB_SIZE>> {
        let t = self.tail.load(core::sync::atomic::Ordering::Relaxed);
        if t.is_null() {
            None
        } else {
            Some(unsafe { &mut *t })
        }
    }

    /// Returns a reference to the first slab owned by self
    fn head(&self) -> Option<&mut Slab<T, SLAB_SIZE>> {
        let t = self.head.load(core::sync::atomic::Ordering::Relaxed);
        if t.is_null() {
            None
        } else {
            Some(unsafe { &mut *t })
        }
    }

    /// Returns a reference to the slab pointed at by the cursor.
    fn cursor(&self) -> Option<&mut Slab<T, SLAB_SIZE>> {
        let t = self.cursor.load(core::sync::atomic::Ordering::Relaxed);
        if t.is_null() {
            None
        } else {
            Some(unsafe { &mut *t })
        }
    }

    /// Updates the cursor to point to `next`.
    /// This method should be called when `self.cursor` failed to allocate memory and the visitor count is zero.
    ///
    /// # Safety
    ///
    /// This method is unsafe because the caller must ensure that `self.lock` is set and there are
    /// no visitors.
    unsafe fn advance_cursor(&self, next: &mut Slab<T, SLAB_SIZE>) {
        debug_assert!(self.lock.load(core::sync::atomic::Ordering::Relaxed));
        debug_assert_eq!(self.visitors.load(core::sync::atomic::Ordering::Relaxed), 0);
        self.cursor
            .store(next, core::sync::atomic::Ordering::Relaxed);
    }
}

unsafe impl<A: core::alloc::Allocator, T, const SLAB_SIZE: usize> core::alloc::Allocator
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

        while self
            .lock
            .compare_exchange_weak(
                false,
                true,
                core::sync::atomic::Ordering::Acquire,
                core::sync::atomic::Ordering::Relaxed,
            )
            .is_err()
        {
            core::hint::spin_loop();
        }
        self.visitors
            .fetch_add(1, core::sync::atomic::Ordering::Acquire);
        self.lock
            .store(false, core::sync::atomic::Ordering::Release);

        let mut dirty = false;
        let mut slab = self.cursor().ok_or(AllocError)?;
        let ret = loop {
            if let Ok(ret) = slab.alloc() {
                break Ok(ret);
            } else {
                dirty = true;
                slab = slab.next_slab().ok_or(AllocError)?;
            }
        };

        while self
            .lock
            .compare_exchange_weak(
                false,
                true,
                core::sync::atomic::Ordering::Acquire,
                core::sync::atomic::Ordering::Relaxed,
            )
            .is_err()
        {
            core::hint::spin_loop();
        }
        if self
            .visitors
            .fetch_sub(1, core::sync::atomic::Ordering::Acquire)
            == 1
            && dirty
        {
            // SAFETY: We have acquired the lock bit, and can assert that the visitor count is 0
            unsafe { self.advance_cursor(slab) }
        }
        self.lock
            .store(false, core::sync::atomic::Ordering::Release);

        ret
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
        let t = self.slab_metadata.alloc().ok_or(AllocError)?
            - SlabMetadata::<T, SLAB_SIZE>::reserved_bits();
        let u = &mut self.obj_elements[t] as *mut core::mem::MaybeUninit<T>;
        // SAFETY: Pointer is taken from a reference
        Ok(unsafe {
            NonNull::new_unchecked(core::slice::from_raw_parts_mut(u.cast(), size_of::<T>()))
        })
    }

    fn free(&self, index: usize) {
        self.slab_metadata.set_bit(index, false);
    }

    fn next_slab(&self) -> Option<&mut Self> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_slab_size() {
        let slab = Slab::<u16, 64>::new();
        //assert_eq!(core::mem::size_of::<Slab<u128, 64>>(), 64);
        assert_eq!(slab.obj_elements.len(), 20);
        let slab = Slab::<u128, 64>::new();
        assert_eq!(core::mem::size_of::<Slab<u128, 64>>(), 64);
        assert_eq!(slab.obj_elements.len(), 2);
    }

    #[test]
    fn obj_elem_count() {
        assert_eq!(slab_count_obj_elements::<u128, 64>(), 2);
        assert_eq!(slab_count_obj_elements::<u16, 64>(), 20);
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

    }
}
