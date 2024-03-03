// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "media/gpu/windows/d3d11_copying_texture_wrapper.h"

#include <memory>

#include "gpu/command_buffer/service/mailbox_manager.h"
#include "media/gpu/windows/d3d11_com_defs.h"

namespace media {

// TODO(tmathmeyer) What D3D11 Resources do we need to do the copying?
CopyingTexture2DWrapper::CopyingTexture2DWrapper(
    const gfx::Size& size,
    std::unique_ptr<Texture2DWrapper> output_wrapper,
    std::unique_ptr<VideoProcessorProxy> processor,
    ComD3D11Texture2D output_texture,
    base::Optional<gfx::ColorSpace> output_color_space)
    : size_(size),
      video_processor_(std::move(processor)),
      output_texture_wrapper_(std::move(output_wrapper)),
      output_texture_(std::move(output_texture)),
      output_color_space_(std::move(output_color_space)) {}

CopyingTexture2DWrapper::~CopyingTexture2DWrapper() = default;

#define RETURN_ON_FAILURE(expr) \
  do {                          \
    if (!SUCCEEDED((expr))) {   \
      return false;             \
    }                           \
  } while (0)

bool CopyingTexture2DWrapper::ProcessTexture(
    ComD3D11Texture2D texture,
    size_t array_slice,
    const gfx::ColorSpace& input_color_space,
    MailboxHolderArray* mailbox_dest,
    gfx::ColorSpace* output_color_space) {
  D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC output_view_desc = {
      D3D11_VPOV_DIMENSION_TEXTURE2D};
  output_view_desc.Texture2D.MipSlice = 0;
  ComD3D11VideoProcessorOutputView output_view;
  RETURN_ON_FAILURE(video_processor_->CreateVideoProcessorOutputView(
      output_texture_.Get(), &output_view_desc, &output_view));

  D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC input_view_desc = {0};
  input_view_desc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
  input_view_desc.Texture2D.ArraySlice = array_slice;
  input_view_desc.Texture2D.MipSlice = 0;
  ComD3D11VideoProcessorInputView input_view;
  RETURN_ON_FAILURE(video_processor_->CreateVideoProcessorInputView(
      texture.Get(), &input_view_desc, &input_view));

  D3D11_VIDEO_PROCESSOR_STREAM streams = {0};
  streams.Enable = TRUE;
  streams.pInputSurface = input_view.Get();

  // If we were given an output color space, then that's what we'll use.
  // Otherwise, we'll use whatever the input space is.
  gfx::ColorSpace copy_color_space =
      output_color_space_ ? *output_color_space_ : input_color_space;

  // If the input color space has changed, or if this is the first call, then
  // notify the video processor about it.
  if (!previous_input_color_space_ ||
      *previous_input_color_space_ != input_color_space) {
    previous_input_color_space_ = input_color_space;
    video_processor_->SetStreamColorSpace(input_color_space);
    video_processor_->SetOutputColorSpace(copy_color_space);
  }

  RETURN_ON_FAILURE(video_processor_->VideoProcessorBlt(output_view.Get(),
                                                        0,  // output_frameno
                                                        1,  // stream_count
                                                        &streams));

  return output_texture_wrapper_->ProcessTexture(
      output_texture_, 0, copy_color_space, mailbox_dest, output_color_space);
}

bool CopyingTexture2DWrapper::Init(GetCommandBufferHelperCB get_helper_cb) {
  if (!video_processor_->Init(size_.width(), size_.height()))
    return false;

  return output_texture_wrapper_->Init(std::move(get_helper_cb));
}

}  // namespace media
