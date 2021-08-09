//
//  Convert.swift
//  iosTensorflowLite
//
//  Created by Yudi Edri Alviska on 06/08/21.
//

import Foundation
import UIKit
import MLKit

public func uiImageToPixelBuffer(image: UIImage, size: Int) -> CVPixelBuffer? {
    let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
    var pixelBuffer : CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, size, size, kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
    guard (status == kCVReturnSuccess) else {
        return nil
    }
    
    CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
    let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
    
    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(data: pixelData, width: size, height: size, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
    
    context?.translateBy(x: 0, y: CGFloat(size))
    context?.scaleBy(x: 1.0, y: -1.0)
    
    UIGraphicsPushContext(context!)
    image.draw(in: CGRect(x: 0, y: 0, width: size, height: size))
    UIGraphicsPopContext()
    CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
    return pixelBuffer
}

public func getImageFace(from sampleBuffer: CMSampleBuffer?, rectImage: CGRect) -> UIImage? {
    guard let sampleBuffer = sampleBuffer else {
        print("Sample buffer is NULL.")
        return nil
    }
    guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
        print("Invalid sample buffer.")
        return nil
    }
    
    CVPixelBufferLockBaseAddress(imageBuffer, CVPixelBufferLockFlags.readOnly)
    
    let baseAddress = CVPixelBufferGetBaseAddress(imageBuffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer)
    let bitPerComponent: size_t = 8 // TODO: This may vary on other formats.
    let width = CVPixelBufferGetWidth(imageBuffer)
    let height = CVPixelBufferGetHeight(imageBuffer)
    
    // TODO: Add more support for non-RGB color space.
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    
    // TODO: Add more support for other formats.
    guard let context = CGContext(data: baseAddress, width: width, height: height, bitsPerComponent: bitPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue) else {
        print("Failed to create CGContextRef")
        CVPixelBufferUnlockBaseAddress(imageBuffer, .readOnly)
        return nil
    }
    
    guard let cgImage = context.makeImage() else {
        print("Failed to create CGImage")
        CVPixelBufferUnlockBaseAddress(imageBuffer, .readOnly)
        return nil
    }
    
    CVPixelBufferUnlockBaseAddress(imageBuffer, .readOnly)
    
    let imageRef: CGImage = cgImage.cropping(to: rectImage)!
    let imageCrop: UIImage = UIImage(cgImage: imageRef, scale: 0.5, orientation: .right)
    return imageCrop
}

public func getImageFaceFromUIImage(from image: UIImage, rectImage: CGRect) -> UIImage? {
    let imageRef: CGImage = (image.cgImage?.cropping(to: rectImage)!)!
    let imageCrop: UIImage = UIImage(cgImage: imageRef, scale: 0.5, orientation: image.imageOrientation)
    return imageCrop
}
