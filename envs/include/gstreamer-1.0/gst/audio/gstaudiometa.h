/* GStreamer
 * Copyright (C) <2011> Wim Taymans <wim.taymans@gmail.com>
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
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#ifndef __GST_AUDIO_META_H__
#define __GST_AUDIO_META_H__

#include <gst/audio/audio.h>

G_BEGIN_DECLS

#define GST_AUDIO_DOWNMIX_META_API_TYPE (gst_audio_downmix_meta_api_get_type())
#define GST_AUDIO_DOWNMIX_META_INFO  (gst_audio_downmix_meta_get_info())

typedef struct _GstAudioDownmixMeta GstAudioDownmixMeta;

/**
 * GstAudioDownmixMeta:
 * @meta: parent #GstMeta
 * @from_position: the channel positions of the source
 * @to_position: the channel positions of the destination
 * @from_channels: the number of channels of the source
 * @to_channels: the number of channels of the destination
 * @matrix: the matrix coefficients.
 *
 * Extra buffer metadata describing audio downmixing matrix. This metadata is
 * attached to audio buffers and contains a matrix to downmix the buffer number
 * of channels to @channels.
 *
 * @matrix is an two-dimensional array of @to_channels times @from_channels
 * coefficients, i.e. the i-th output channels is constructed by multiplicating
 * the input channels with the coefficients in @matrix[i] and taking the sum
 * of the results.
 */
struct _GstAudioDownmixMeta {
  GstMeta      meta;

  GstAudioChannelPosition *from_position;
  GstAudioChannelPosition *to_position;
  gint        from_channels, to_channels;
  gfloat       **matrix;
};

GST_AUDIO_API
GType gst_audio_downmix_meta_api_get_type (void);

GST_AUDIO_API
const GstMetaInfo * gst_audio_downmix_meta_get_info (void);

#define gst_buffer_get_audio_downmix_meta(b) ((GstAudioDownmixMeta*)gst_buffer_get_meta((b), GST_AUDIO_DOWNMIX_META_API_TYPE))
GST_AUDIO_API
GstAudioDownmixMeta * gst_buffer_get_audio_downmix_meta_for_channels    (GstBuffer *buffer,
                                                                         const GstAudioChannelPosition *to_position,
                                                                         gint                           to_channels);

GST_AUDIO_API
GstAudioDownmixMeta * gst_buffer_add_audio_downmix_meta (GstBuffer    *buffer,
                                                         const GstAudioChannelPosition *from_position,
                                                         gint                           from_channels,
                                                         const GstAudioChannelPosition *to_position,
                                                         gint                           to_channels,
                                                         const gfloat                 **matrix);


#define GST_AUDIO_CLIPPING_META_API_TYPE (gst_audio_clipping_meta_api_get_type())
#define GST_AUDIO_CLIPPING_META_INFO  (gst_audio_clipping_meta_get_info())

typedef struct _GstAudioClippingMeta GstAudioClippingMeta;

/**
 * GstAudioClippingMeta:
 * @meta: parent #GstMeta
 * @format: GstFormat of @start and @stop, GST_FORMAT_DEFAULT is samples
 * @start: Amount of audio to clip from start of buffer
 * @end: Amount of  to clip from end of buffer
 *
 * Extra buffer metadata describing how much audio has to be clipped from
 * the start or end of a buffer. This is used for compressed formats, where
 * the first frame usually has some additional samples due to encoder and
 * decoder delays, and the last frame usually has some additional samples to
 * be able to fill the complete last frame.
 *
 * This is used to ensure that decoded data in the end has the same amount of
 * samples, and multiply decoded streams can be gaplessly concatenated.
 *
 * Note: If clipping of the start is done by adjusting the segment, this meta
 * has to be dropped from buffers as otherwise clipping could happen twice.
 *
 * Since: 1.8
 */
struct _GstAudioClippingMeta {
  GstMeta   meta;

  GstFormat format;
  guint64   start;
  guint64   end;
};

GST_AUDIO_API
GType gst_audio_clipping_meta_api_get_type (void);

GST_AUDIO_API
const GstMetaInfo * gst_audio_clipping_meta_get_info (void);

#define gst_buffer_get_audio_clipping_meta(b) ((GstAudioClippingMeta*)gst_buffer_get_meta((b), GST_AUDIO_CLIPPING_META_API_TYPE))

GST_AUDIO_API
GstAudioClippingMeta * gst_buffer_add_audio_clipping_meta (GstBuffer *buffer,
                                                           GstFormat  format,
                                                           guint64    start,
                                                           guint64    end);

G_END_DECLS

#endif /* __GST_AUDIO_META_H__ */
