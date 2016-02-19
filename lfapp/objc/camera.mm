/*
 * $Id: lfobjc.mm 1299 2011-12-11 21:39:12Z justin $
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

#ifdef LFL_CAMERA

extern "C" void UpdateFrame(const char *imageData, int width, int height, int ImageSize);

#if defined(LFL_IPHONE) && !defined(LFL_IPHONESIM) /* no AVCapture on simulator */

/* AVCapture */
#import <CoreGraphics/CoreGraphics.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>

@interface AVCapture : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate> { }
  - (id) init;
  - (void) initCamera;
 
  + (AVCapture *) instance;
  + (void) shutdown;
   
@end

@implementation AVCapture
  static AVCapture *gAVCapture = nil;
  static AVCaptureSession *gAVCaptureSession = nil;
  static int gCameraFPS = 0;

  - (void) dealloc {
    fprintf(stderr, "AVCapture dealloc\n");
    if (gAVCaptureSession != nil) { [gAVCaptureSession release]; gAVCaptureSession = nil; }
    [super dealloc];
  }
  
  - (id) init {
    self = [super init];
    if (!self) return nil;
    [self initCamera];
    fprintf(stderr, "AVCapture init\n");
    return self;
  }
  
  + (void) shutdown {
    fprintf(stderr, "AVCapture shutdown\n");
    if (gAVCapture != nil) { [gAVCapture release]; gAVCapture = nil; }
  }
 
  + (AVCapture *) instance {
    /* global capture object - use singleton class to access/shutdown */
    if (gAVCapture == nil) {
      fprintf(stderr, "AVCapture alloc/init instance\n");
      gAVCapture = [[[AVCapture alloc] init] retain];
    }
    return gAVCapture;
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
    if (gAVCaptureSession != nil) { [gAVCaptureSession release]; gAVCaptureSession = nil; }
 
    // Create a CaptureSession
    gAVCaptureSession = [[[AVCaptureSession alloc] init] retain];
 
    // Add to CaptureSession input and output
    [gAVCaptureSession addInput:captureInput];
    [gAVCaptureSession addOutput:captureOutput];
 
    // Configure CaptureSession Quality
    [gAVCaptureSession beginConfiguration];
    //#define AVCAPTURE_HIGH_QUALITY
#ifdef AVCAPTURE_HIGH_QUALITY
    [gAVCaptureSession setSessionPreset:AVCaptureSessionPresetHigh];
#else
    [gAVCaptureSession setSessionPreset:AVCaptureSessionPresetLow];
#endif
    [gAVCaptureSession commitConfiguration];

    NSError *error = nil;
    if([captureDevice lockForConfiguration:&error]) {
      captureDevice.torchMode = AVCaptureTorchModeOn;
      [captureDevice unlockForConfiguration];
    }
    // Start the capture
    [gAVCaptureSession startRunning];

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
    UpdateFrame((char *)baseAddress, width, height, width * height * 4); 

    /* unlock pixel buffer */
    CVPixelBufferUnlockBaseAddress(imageBuffer,0);

#ifdef AVCAPTURE_SERIAL_QUEUE
    [pool drain];
#endif
} 
@end

void AVCaptureInit(int fps) {
  AVCapture *avCapture = [AVCapture instance];
  [UIApplication sharedApplication].statusBarHidden = YES;
}

void AVCaptureShutdown() {
 	[AVCapture shutdown];
}
#endif // #defined(LFL_IPHONE) && !defined(LFL_IPHONESIM)

#if __x86_64__ || __ppc64__ || __amd64__
 #define LFL64
#else
 #define LFL32
#endif

#ifdef LFL64
#import <Cocoa/Cocoa.h>
#import <QTKit/QTKit.h>

@interface QTKitCamera : NSObject
  {
    QTCaptureSession                 *mCaptureSession;
    QTCaptureDeviceInput             *mCaptureVideoDeviceInput;
    QTCaptureDecompressedVideoOutput *mCaptureDecompressedVideoOutput;
    IBOutlet QTCaptureView           *mCaptureView;
  }

  + (QTKitCamera *) instance;
  - (void) shutdown;
  - (id) init;
@end

@implementation QTKitCamera
  static QTKitCamera *gQTKitCamera = nil;

  + (QTKitCamera *) instance {
    if (gQTKitCamera == nil) {
      fprintf(stderr, "QTKitCamera alloc/init instance\n");
      gQTKitCamera = [[[QTKitCamera alloc] init] retain];
    }
    return gQTKitCamera;
  }

  - (void) shutdown {
    fprintf(stderr, "QTKitCamera shutdown\n");
    [mCaptureSession stopRunning];
    [[mCaptureVideoDeviceInput device] close];
  }

  - (id) init {
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
    UpdateFrame(reinterpret_cast<char*>(baseAddress), width, height, rowbytes * height); 

    /* unlock pixel buffer */
    CVPixelBufferUnlockBaseAddress(pixels,0);
    CVBufferRelease(pixels);
  }
@end

void QTKitInit(int fps) {
  QTKitCamera *qtkitCamera = [QTKitCamera instance];
}

void QTKitShutdown() {
  QTKitCamera *qtkitCamera = [QTKitCamera instance];
  [qtkitCamera shutdown];
}
#endif // LFL64

#endif /* LFL_CAMERA */
