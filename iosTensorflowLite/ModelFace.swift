//
//  ModelFace.swift
//  iosTensorflowLite
//
//  Created by Yudi Edri Alviska on 06/08/21.
//

import Foundation
import UIKit
import TensorFlowLite

class ModelFace {
    var id: String?
    var title: String?
    var distance: Float?
    var extra: Tensor?
    var location: CGRect?
    var color: UIColor?
    var crop: UIImage?
    
    init(id: String, title: String, distance: Float, location: CGRect) {
        self.id = id
        self.title = title
        self.distance = distance
        self.location = location
    }
    
    func setId(id: String) {
        self.id = id
    }
    
    func setTitle(title: String) {
        self.title = title
    }
    
    func setDistance(distance: Float) {
        self.distance = distance
    }
    
    func setExtra(extra: Tensor) {
        self.extra = extra
    }
    
    func setLocation(location: CGRect) {
        self.location = location
    }
    
    func setColor(color: UIColor) {
        self.color = color
    }
    
    func setCrop(crop: UIImage) {
        self.crop = crop
    }
    
    func getId() -> String? {
        return self.id
    }
    
    func getTitle() -> String? {
        return self.title
    }
    
    func getDistance() -> Float? {
        return self.distance
    }
    
    func getExtra() -> Tensor? {
        return self.extra
    }
    
    func getLocation() -> CGRect? {
        return self.location
    }
    
    func getColor() -> UIColor? {
        return self.color
    }
    
    func getCrop() -> UIImage? {
        return self.crop
    }
}
