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

#import <CoreGraphics/CoreGraphics.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>
typedef std::function<void(const char*, int, int, int)> UpdateFrameCB;

@interface AVCapture : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate> { }
  {
    AVCaptureSession *mAVCaptureSession;
    UpdateFrameCB     frame_cb;
  }
  - (id) initWithFrameCB: (UpdateFrameCB)cb;
  - (void) initCamera;
  + (void) shutdown;
@end

@implementation AVCapture
  - (id) initWithFrameCB: (UpdateFrameCB)cb {
    self = [super init];
    if (!self) return nil;
    frame_cb = cb;
    [self initCamera];
    fprintf(stderr, "AVCapture init\n");
    return self;
  }
  
  + (void) shutdown {
    fprintf(stderr, "AVCapture shutdown\n");
    if (mCaptureSession != nil) { [mAVCaptureSession release]; mAVCaptureSession = nil; }
  }

  - (void) dealloc {
    fprintf(stderr, "AVCapture dealloc\n");
    [super dealloc];
  }
 
  /* 
  init camera code pieced together frm forum + blog post:
  http://forum.unity3d.com/threads/46322-Unity-iPhone-and-Capture-video
  http://www.benjaminloulier.com/posts/2-ios4-and-direct-access-to-the-camera 
  http://www.iphonedevsdk.com/forum/iphone-sdk-development/52467-avfoundation-troubles.html
  */
  - (void) initCamera {
    NSArray *devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    for (int i=0; i < [devices count]; i++)
      fprintf(stderr, "AVCapture camera: %s\n", [[[devices objectAtIndex:i] description] cString]);
  
    AVCaptureDevice *captureDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    fprintf(stderr, "AVCapture using camera %s (position %d)\n", [[captureDevice uniqueID] cString], [captureDevice position]);
 
    // Create a capture input
    AVCaptureDeviceInput *captureInput = [AVCaptureDeviceInput deviceInputWithDevice:captureDevice error:nil];
 
    // Create a capture output
    AVCaptureVideoDataOutput *captureOutput = [[[AVCaptureVideoDataOutput alloc] init] autorelease];
    // fps
    captureOutput.minFrameDuration = CMTimeMake(1, 15);
 
#define AVCAPTURE_ALWAYS_DISCARD_LATE_FRAMES
#ifdef AVCAPTURE_ALWAYS_DISCARD_LATE_FRAMES
    /*While a frame is processes in -captureOutput:didOutputSampleBuffer:fromConnection: delegate methods no other frames are added in the queue.  If you don't want this behaviour set the property to NO */
    captureOutput.alwaysDiscardsLateVideoFrames = YES;
#endif
 
#define AVCAPTURE_SERIAL_QUEUE /* create serial queue to handle the processing of our frames*/
#define AVCAPTURE_DEDICATED_CAMERA_QUEUE /* dedicated queue to create and release */

#ifdef AVCAPTURE_SERIAL_QUEUE
#ifdef AVCAPTURE_DEDICATED_CAMERA_QUEUE
    dispatch_queue_t queue = dispatch_queue_create("cameraQueue", NULL);
    [captureOutput setSampleBufferDelegate:self queue:queue];
    dispatch_release(queue);
#else
    /* best practice */
    [captureOutput setSampleBufferDelegate:self queue:dispatch_get_main_queue()];
#endif
#else /* no serial queue */
    [captureOutput setSampleBufferDelegate:self queue:NULL];
#endif
 
    // Set the video output to store frame in BGRA (It is supposed to be faster)
    [captureOutput setVideoSettings:[NSDictionary dictionaryWithObject:
      [NSNumber numberWithInt:kCVPixelFormatType_32BGRA] 
      forKey:(id)kCVPixelBufferPixelFormatTypeKey]];
 
    // Clear existing CaptureSession
    if (mAVCaptureSession != nil) { [mAVCaptureSession release]; mAVCaptureSession = nil; }
 
    // Create a CaptureSession
    mAVCaptureSession = [[[AVCaptureSession alloc] init] retain];
 
    // Add to CaptureSession input and output
    [mAVCaptureSession addInput:captureInput];
    [mAVCaptureSession addOutput:captureOutput];
 
    // Configure CaptureSession Quality
    [mAVCaptureSession beginConfiguration];
    //#define AVCAPTURE_HIGH_QUALITY
#ifdef AVCAPTURE_HIGH_QUALITY
    [mAVCaptureSession setSessionPreset:AVCaptureSessionPresetHigh];
#else
    [mAVCaptureSession setSessionPreset:AVCaptureSessionPresetLow];
#endif
    [mAVCaptureSession commitConfiguration];

    NSError *error = nil;
    if([captureDevice lockForConfiguration:&error]) {
      captureDevice.torchMode = AVCaptureTorchModeOn;
      [captureDevice unlockForConfiguration];
    }
    // Start the capture
    [mAVCaptureSession startRunning];

    fprintf(stderr, "AVCapture started\n");
  }
 
  - (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
#ifdef AVCAPTURE_SERIAL_QUEUE
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
#endif
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVPixelBufferLockBaseAddress(imageBuffer,0);

    /* image pixel dimensions */
    //#define AVCAPTURE_PIXEL_BUFFER_PLANES
#ifdef AVCAPTURE_PIXEL_BUFFER_PLANES
    uint8_t *baseAddress = (uint8_t *)CVPixelBufferGetBaseAddressOfPlane(imageBuffer, 0);
#else
    uint8_t *baseAddress = (uint8_t *)CVPixelBufferGetBaseAddress(imageBuffer);
#endif
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);

    /* write data to ring buffer */
    frame_cb((char *)baseAddress, width, height, width * height * 4); 

    /* unlock pixel buffer */
    CVPixelBufferUnlockBaseAddress(imageBuffer,0);

#ifdef AVCAPTURE_SERIAL_QUEUE
    [pool drain];
#endif
} 
@end

namespace LFL {
struct AVCaptureCameraModule : public Module {
  mutex lock; 
  struct Stream {
    unique_ptr<RingSampler> frames;
    int next=0;
  } L, R;
  CameraState *state;
  AVCapture *camera=0;
  AVCaptureCameraModule(CameraState *S) : state(S) {} 

  int Init() {
    camera = [[[AVCapture alloc] initWithFrameCB:
      bind(&AVCaptureCameraModule::UpdateFrame, this, _1, _2, _3, _4)] retain];
    [UIApplication sharedApplication].statusBarHidden = YES;

    /* hardcoded for now per AVCaptureSessionPresetLow */
    FLAGS_camera_image_height = 144;
    FLAGS_camera_image_width = 192;
    state->image_format = Pixel::BGR32;
    state->image_linesize = FLAGS_camera_image_width*4;
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
    state->image = (unsigned char*)L.frames->Read(-1, L.next);
    state->image_timestamp = L.frames->ReadTimestamp(-1, L.next);
    return 1;
  }

  void UpdateFrame(const char *imageData, int width, int height, int imageSize) {
    if (!L.frames) L.frames = make_unique<RingSampler>(FLAGS_camera_fps, FLAGS_camera_fps, imageSize);
    memcpy(L.frames->Write(RingSampler::Peek | RingSampler::Stamp), imageData, imageSize);
    { /* commit */  
      ScopedMutex ML(lock);
      L.frames->Write();
      state->frames_read++;
    } 
  }
};

extern "C" void *LFAppCreateCameraModule(CameraState *state) { return new AVCaptureCameraModule(state); }

}; // namespace LFL
