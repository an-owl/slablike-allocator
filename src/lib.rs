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

use core::ptr::NonNull;
use slab_meta::SlabMetadata;

pub use slab_meta::meta_bitmap_size;

pub struct SlabLike<A: core::alloc::Allocator, T, const SLAB_SIZE: usize>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    alloc: A,

    head: Option<NonNull<Slab<T, SLAB_SIZE>>>,
    tail: Option<NonNull<Slab<T, SLAB_SIZE>>>,
    cursor: Option<NonNull<Slab<T, SLAB_SIZE>>>,
}

impl<A: core::alloc::Allocator, T, const SLAB_SIZE: usize> SlabLike<A, T, SLAB_SIZE>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    pub const fn new(alloc: A) -> Self {
        Self {
            alloc,
            head: None,
            tail: None,
            cursor: None,
        }
    }

    /// Allocates a new slab, appends it to the end of the linked list
    fn new_slab(&mut self) -> Result<(), AllocError> {
        let ptr = self.alloc.allocate(Layout::new::<Slab<T, SLAB_SIZE>>())?;
        let n_slab: &mut Slab<T, SLAB_SIZE> = unsafe { &mut *ptr.as_ptr().cast() };

        // First slab, initialize all pointers
        if self.head.is_none() {
            self.head = Some(NonNull::from(n_slab));
            self.tail = Some(NonNull::from(n_slab));
            self.cursor = Some(NonNull::from(n_slab));
            return Ok(());
        }

        self.tail
    }

    /// Returns a reference to the last slab owned by self.
    fn tail(&mut self) -> Option<&mut Slab<T, SLAB_SIZE>> {
        Some(unsafe { &mut *self.tail?.as_ptr() })
    }

    /// Returns a reference to the first slab owned by self
    fn head(&mut self) -> Option<&mut Slab<T, SLAB_SIZE>> {
        Some(unsafe { &mut *self.head?.as_ptr() })
    }

    /// Returns a reference to the slab pointed at by the cursor.
    fn cursor(&mut self) -> Option<&mut Slab<T, SLAB_SIZE>> {
        Some(unsafe { &mut *self.cursor?.as_ptr() })
    }
}

unsafe impl<A: core::alloc::Allocator, T, const SLAB_SIZE: usize> core::alloc::Allocator
    for SlabLike<A, T, SLAB_SIZE>
{
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        assert_eq!(
            layout,
            Layout::new::<T>(),
            "Unexpected layout for {}: {layout:?}, expected {:?}",
            core::any::type_name::<Self>(),
            Layout::new::<T>()
        );
    }
}

/// Returns the number of object elements that can be stored in a [Slab]
///
/// This fn will panic if the number of object elements is 0.
pub const fn slab_count_obj_elements<T, const SLAB_SIZE: usize>() -> usize {
    let elem_size = size_of::<[T; 2]>() / 2;
    let raw_obj_elem_count = SLAB_SIZE / elem_size;

    let meta_size = slab_meta::size_of_meta::<T>(SLAB_SIZE);
    let meta_obj_elem_size = meta_size / elem_size;

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

    fn alloc(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert_eq!(layout, Layout::new::<T>());
        let t = self.slab_metadata.alloc().ok_or(AllocError)?;
        let u = &mut self.obj_elements[t] as *mut T;
        // SAFETY: Pointer is taken from a reference
        Ok(unsafe { NonNull::new_unchecked(u.cast()) })
    }

    fn next_slab(&mut self) -> Option<&mut Self> {
        Some(unsafe { &mut *self.slab_metadata.next_slab()? })
    }

    fn prev_slab(&mut self) -> Option<&mut Self> {
        Some(unsafe { &mut *self.slab_metadata.prev_slab()? })
    }

    fn set_next(&mut self, slab: Option<&mut Self>) {
        self.slab_metadata.set_next(slab.map(|p| p as *mut _));
    }
    fn set_prev(&mut self, slab: Option<&mut Self>) {
        self.slab_metadata.set_prev(slab.map(|p| p as *mut _));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_slab_size() {
        let slab = Slab::<u128, 64>::new();
        assert_eq!(slab.obj_elements.len(), 2);
    }
}
