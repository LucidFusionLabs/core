/*
 * $Id: camera.cpp 1330 2014-11-06 03:04:15Z justin $
 * Copyright (C) 2009 Lucid Fusion Labs

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#import <Cocoa/Cocoa.h>
#import <QTKit/QTKit.h>
#include "core/app/app.h"
typedef std::function<void(const char*, int, int, int)> UpdateFrameCB;

@interface QTKitCamera : NSObject
  {
    QTCaptureSession                 *mCaptureSession;
    QTCaptureDeviceInput             *mCaptureVideoDeviceInput;
    QTCaptureDecompressedVideoOutput *mCaptureDecompressedVideoOutput;
    IBOutlet QTCaptureView           *mCaptureView;
    UpdateFrameCB                     frame_cb;
  }
  - (id) initWithFrameCB: (UpdateFrameCB)cb;
  - (void) shutdown;
@end

@implementation QTKitCamera
  - (id) initWithFrameCB: (UpdateFrameCB)cb {
    self = [super init];
    frame_cb = cb;

    NSError *error = nil;
    QTCaptureDevice *videoDevice = [QTCaptureDevice defaultInputDeviceWithMediaType:QTMediaTypeVideo];
    if (![videoDevice open:&error]) {
      fprintf(stderr,  "videoDevice open\n");
      return nil;
    }
  
    mCaptureSession = [[QTCaptureSession alloc] init];
    mCaptureVideoDeviceInput = [[QTCaptureDeviceInput alloc] initWithDevice:videoDevice];
    if (![mCaptureSession addInput:mCaptureVideoDeviceInput error:&error]) {
      fprintf(stderr, "mCaptureSession addInput open\n");
      return nil;
    }

    NSDictionary *pixelBufferOptions = [NSDictionary dictionaryWithObjectsAndKeys:
                                       [NSNumber numberWithUnsignedInt:kCVPixelFormatType_32BGRA],
                                       id(kCVPixelBufferPixelFormatTypeKey),
                                       nil];

    mCaptureDecompressedVideoOutput = [[QTCaptureDecompressedVideoOutput alloc] init];
    [mCaptureDecompressedVideoOutput setDelegate:self];
    [mCaptureDecompressedVideoOutput setPixelBufferAttributes:pixelBufferOptions];
    if (![mCaptureSession addOutput:mCaptureDecompressedVideoOutput error:&error]) {
      fprintf(stderr, "mCaptureSession addOutput open\n");
      return nil;
    }

    [mCaptureSession startRunning];
    return self;
  }

  - (void)captureOutput:(QTCaptureOutput *)captureOutput
    didOutputVideoFrame:(CVImageBufferRef)imageBuffer
       withSampleBuffer:(QTSampleBuffer *)sampleBuffer
         fromConnection:(QTCaptureConnection *)connection
  {
    CVPixelBufferRef pixels = CVBufferRetain(imageBuffer);
    CVPixelBufferLockBaseAddress(pixels,0);

    /* image pixel dimensions */
    uint8_t *baseAddress = reinterpret_cast<uint8_t*>(CVPixelBufferGetBaseAddress(pixels));
    size_t width = CVPixelBufferGetWidth(pixels);
    size_t height = CVPixelBufferGetHeight(pixels);
    size_t rowbytes = CVPixelBufferGetBytesPerRow(pixels);

    /* write data to ring buffer */
    frame_cb(reinterpret_cast<char*>(baseAddress), width, height, rowbytes * height); 

    /* unlock pixel buffer */
    CVPixelBufferUnlockBaseAddress(pixels,0);
    CVBufferRelease(pixels);
  }

  - (void) shutdown {
    fprintf(stderr, "QTKitCamera shutdown\n");
    [mCaptureSession stopRunning];
    [[mCaptureVideoDeviceInput device] close];
  }
@end

namespace LFL {
struct QTKitCameraModule : public Module {
  mutex lock; 
  struct Stream {
    unique_ptr<RingSampler> frames;
    int next=0;
  } L, R;
  CameraState *state;
  QTKitCamera *camera=0;
  QTKitCameraModule(CameraState *S) : state(S) {} 

  int Init() {
    camera = [[[QTKitCamera alloc] initWithFrameCB:
      bind(&QTKitCameraModule::UpdateFrame, this, _1, _2, _3, _4)] retain];
    return 0;
  }

  int Free() {
    [camera shutdown];
    return 0;
  }

  int Frame(unsigned) {
    bool new_frame = false;
    {
      ScopedMutex ML(lock);
      if (state->frames_read > state->last_frames_read) {
        if (L.frames) L.next = L.frames->ring.back;
        if (R.frames) R.next = R.frames->ring.back;
        new_frame = true;
      }
      state->last_frames_read = state->frames_read;
    }
    if (!new_frame) return 0;
    state->image = static_cast<unsigned char *>(L.frames->Read(-1, L.next));
    state->image_timestamp_us = L.frames->ReadTimestamp(-1, L.next).count();
    return 1;
  }

  void UpdateFrame(const char *imageData, int width, int height, int imageSize) {
    if (!L.frames) {
      L.frames = make_unique<RingSampler>(FLAGS_camera_fps, FLAGS_camera_fps, imageSize);
      FLAGS_camera_image_width = width;
      FLAGS_camera_image_height = height;
      state->image_format = Pixel::RGB32;
      state->image_linesize = FLAGS_camera_image_width*4;
    }
    memcpy(L.frames->Write(RingSampler::Peek | RingSampler::Stamp), imageData, imageSize);
    { /* commit */  
      ScopedMutex ML(lock);
      L.frames->Write();
      state->frames_read++;
    } 
  }
};

unique_ptr<Module> CreateCameraModule(CameraState *state) { return make_unique<QTKitCameraModule>(state); }

}; // namespace LFL
