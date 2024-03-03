// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <utility>

#include "base/bind_helpers.h"
#include "media/gpu/windows/d3d11_copying_texture_wrapper.h"
#include "media/gpu/windows/d3d11_texture_wrapper.h"
#include "media/gpu/windows/d3d11_video_processor_proxy.h"
#include "testing/gmock/include/gmock/gmock.h"
#include "testing/gtest/include/gtest/gtest.h"

using ::testing::_;
using ::testing::Bool;
using ::testing::Combine;
using ::testing::Return;
using ::testing::Values;

namespace media {

class MockVideoProcessorProxy : public VideoProcessorProxy {
 public:
  MockVideoProcessorProxy() : VideoProcessorProxy(nullptr, nullptr) {}

  bool Init(uint32_t width, uint32_t height) override {
    return MockInit(width, height);
  }

  HRESULT CreateVideoProcessorOutputView(
      ID3D11Texture2D* output_texture,
      D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC* output_view_descriptor,
      ID3D11VideoProcessorOutputView** output_view) override {
    return MockCreateVideoProcessorOutputView();
  }

  HRESULT CreateVideoProcessorInputView(
      ID3D11Texture2D* input_texture,
      D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC* input_view_descriptor,
      ID3D11VideoProcessorInputView** input_view) override {
    return MockCreateVideoProcessorInputView();
  }

  void SetStreamColorSpace(const gfx::ColorSpace& color_space) override {
    last_stream_color_space_ = color_space;
  }

  void SetOutputColorSpace(const gfx::ColorSpace& color_space) override {
    last_output_color_space_ = color_space;
  }

  HRESULT VideoProcessorBlt(ID3D11VideoProcessorOutputView* output_view,
                            UINT output_frameno,
                            UINT stream_count,
                            D3D11_VIDEO_PROCESSOR_STREAM* streams) override {
    return MockVideoProcessorBlt();
  }

  MOCK_METHOD2(MockInit, bool(uint32_t, uint32_t));
  MOCK_METHOD0(MockCreateVideoProcessorOutputView, HRESULT());
  MOCK_METHOD0(MockCreateVideoProcessorInputView, HRESULT());
  MOCK_METHOD0(MockVideoProcessorBlt, HRESULT());

  // Most recent arguments to SetStream/OutputColorSpace().
  base::Optional<gfx::ColorSpace> last_stream_color_space_;
  base::Optional<gfx::ColorSpace> last_output_color_space_;
};

class MockTexture2DWrapper : public Texture2DWrapper {
 public:
  MockTexture2DWrapper() {}

  bool ProcessTexture(ComD3D11Texture2D texture,
                      size_t array_slice,
                      const gfx::ColorSpace& input_color_space,
                      MailboxHolderArray* mailbox_dest,
                      gfx::ColorSpace* output_color_space) override {
    // Pretend we created an arbitrary color space, so that we're sure that it
    // is returned from the copying wrapper.
    *output_color_space = gfx::ColorSpace::CreateHDR10();
    return MockProcessTexture();
  }

  bool Init(GetCommandBufferHelperCB get_helper_cb) override {
    return MockInit();
  }

  MOCK_METHOD0(MockInit, bool());
  MOCK_METHOD0(MockProcessTexture, bool());
};

CommandBufferHelperPtr UselessHelper() {
  return nullptr;
}

class D3D11CopyingTexture2DWrapperTest
    : public ::testing::TestWithParam<
          std::tuple<HRESULT, HRESULT, HRESULT, bool, bool, bool, bool>> {
 public:
#define FIELD(TYPE, NAME, INDEX) \
  TYPE Get##NAME() { return std::get<INDEX>(GetParam()); }
  FIELD(HRESULT, CreateVideoProcessorOutputView, 0)
  FIELD(HRESULT, CreateVideoProcessorInputView, 1)
  FIELD(HRESULT, VideoProcessorBlt, 2)
  FIELD(bool, ProcessorProxyInit, 3)
  FIELD(bool, TextureWrapperInit, 4)
  FIELD(bool, ProcessTexture, 5)
  FIELD(bool, PassthroughColorSpace, 6)
#undef FIELD

  std::unique_ptr<MockVideoProcessorProxy> ExpectProcessorProxy() {
    auto result = std::make_unique<MockVideoProcessorProxy>();
    ON_CALL(*result.get(), MockInit(_, _))
        .WillByDefault(Return(GetProcessorProxyInit()));

    ON_CALL(*result.get(), MockCreateVideoProcessorOutputView())
        .WillByDefault(Return(GetCreateVideoProcessorOutputView()));

    ON_CALL(*result.get(), MockCreateVideoProcessorInputView())
        .WillByDefault(Return(GetCreateVideoProcessorInputView()));

    ON_CALL(*result.get(), MockVideoProcessorBlt())
        .WillByDefault(Return(GetVideoProcessorBlt()));

    return result;
  }

  std::unique_ptr<Texture2DWrapper> ExpectTextureWrapper() {
    auto result = std::make_unique<MockTexture2DWrapper>();

    ON_CALL(*result.get(), MockInit())
        .WillByDefault(Return(GetTextureWrapperInit()));

    ON_CALL(*result.get(), MockProcessTexture())
        .WillByDefault(Return(GetProcessTexture()));

    return std::move(result);
  }

  GetCommandBufferHelperCB CreateMockHelperCB() {
    return base::BindRepeating(&UselessHelper);
  }

  bool InitSucceeds() {
    return GetProcessorProxyInit() && GetTextureWrapperInit();
  }

  bool ProcessTextureSucceeds() {
    return GetProcessTexture() &&
           SUCCEEDED(GetCreateVideoProcessorOutputView()) &&
           SUCCEEDED(GetCreateVideoProcessorInputView()) &&
           SUCCEEDED(GetVideoProcessorBlt());
  }
};

INSTANTIATE_TEST_CASE_P(CopyingTexture2DWrapperTest,
                        D3D11CopyingTexture2DWrapperTest,
                        Combine(Values(S_OK, E_FAIL),
                                Values(S_OK, E_FAIL),
                                Values(S_OK, E_FAIL),
                                Bool(),
                                Bool(),
                                Bool(),
                                Bool()));

// For ever potential return value combination for the D3D11VideoProcessor,
// make sure that any failures result in a total failure.
TEST_P(D3D11CopyingTexture2DWrapperTest,
       CopyingTextureWrapperProcessesCorrectly) {
  gfx::Size size;
  auto processor = ExpectProcessorProxy();
  MockVideoProcessorProxy* processor_raw = processor.get();
  // Provide an unlikely color space, to see if it gets to the video processor,
  // if we're not just doing a pass-through of the input.
  base::Optional<gfx::ColorSpace> copy_color_space;
  if (!GetPassthroughColorSpace())
    copy_color_space = gfx::ColorSpace::CreateDisplayP3D65();
  auto wrapper = std::make_unique<CopyingTexture2DWrapper>(
      size, ExpectTextureWrapper(), std::move(processor), nullptr,
      copy_color_space);

  MailboxHolderArray mailboxes;
  gfx::ColorSpace input_color_space = gfx::ColorSpace::CreateSCRGBLinear();
  gfx::ColorSpace output_color_space;
  EXPECT_EQ(wrapper->Init(CreateMockHelperCB()), InitSucceeds());
  EXPECT_EQ(wrapper->ProcessTexture(nullptr, 0, input_color_space, &mailboxes,
                                    &output_color_space),
            ProcessTextureSucceeds());

  if (ProcessTextureSucceeds()) {
    // Regardless of what the input space is, the output should be provided by
    // the mock wrapper.
    EXPECT_EQ(gfx::ColorSpace::CreateHDR10(), output_color_space);

    // Also expect that the input and copy spaces were provided to the video
    // processor as the stream and output color spaces, respectively.  If no
    // copy space was provided, then expect that the output is the input.
    EXPECT_TRUE(processor_raw->last_stream_color_space_);
    EXPECT_EQ(*processor_raw->last_stream_color_space_, input_color_space);
    EXPECT_TRUE(processor_raw->last_output_color_space_);
    EXPECT_EQ(*processor_raw->last_output_color_space_,
              copy_color_space ? *copy_color_space : input_color_space);
  }

  // TODO: verify that these aren't sent multiple times, unless they change.
}

}  // namespace media
