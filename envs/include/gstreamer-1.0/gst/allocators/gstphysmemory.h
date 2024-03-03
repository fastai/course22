/* GStreamer
 * Copyright (C) 2017 Sebastian Dröge <sebastian@centricular.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef __GST_PHYS_MEMORY_H__
#define __GST_PHYS_MEMORY_H__

#include <gst/gst.h>
#include <gst/allocators/allocators-prelude.h>

G_BEGIN_DECLS

typedef struct _GstPhysMemoryAllocator GstPhysMemoryAllocator;
typedef struct _GstPhysMemoryAllocatorInterface GstPhysMemoryAllocatorInterface;

#define GST_TYPE_PHYS_MEMORY_ALLOCATOR                  (gst_phys_memory_allocator_get_type())
#define GST_IS_PHYS_MEMORY_ALLOCATOR(obj)               (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_PHYS_MEMORY_ALLOCATOR))
#define GST_IS_PHYS_MEMORY_ALLOCATOR_INTERFACE(iface)   (G_TYPE_CHECK_INTERFACE_TYPE ((iface), GST_TYPE_PHYS_MEMORY_ALLOCATOR))
#define GST_PHYS_MEMORY_ALLOCATOR_GET_INTERFACE(obj)    (G_TYPE_INSTANCE_GET_INTERFACE ((obj), GST_TYPE_PHYS_MEMORY_ALLOCATOR, GstPhysMemoryAllocatorInterface))
#define GST_PHYS_MEMORY_ALLOCATOR(obj)                  (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_PHYS_MEMORY_ALLOCATOR, GstPhysMemoryAllocator))
#define GST_PHYS_MEMORY_ALLOCATOR_INTERFACE(iface)      (G_TYPE_CHECK_INTERFACE_CAST ((iface), GST_TYPE_PHYS_MEMORY_ALLOCATOR, GstPhysMemoryAllocatorInterface))
#define GST_PHYS_MEMORY_ALLOCATOR_CAST(obj)             ((GstPhysMemoryAllocator *)(obj))

/**
 * GstPhysMemoryAllocatorInterface:
 *
 * Marker interface for allocators with physical address backed memory
 *
 * Since: 1.14
 */
struct _GstPhysMemoryAllocatorInterface
{
  GTypeInterface parent_iface;

  guintptr (*get_phys_addr) (GstPhysMemoryAllocator * allocator, GstMemory * mem);
};

GST_ALLOCATORS_API
GType gst_phys_memory_allocator_get_type (void);

GST_ALLOCATORS_API
gboolean gst_is_phys_memory (GstMemory *mem);

GST_ALLOCATORS_API
guintptr gst_phys_memory_get_phys_addr (GstMemory * mem);

G_END_DECLS

#endif /* __GST_PHYS_MEMORY_H__ */
