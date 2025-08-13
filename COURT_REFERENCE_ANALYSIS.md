# Tennis Court Reference System - Detailed Analysis

## Overview
The `CourtReference` class defines a standardized coordinate system for tennis court keypoints used in the TennisCourtDetector project. This serves as the ground truth reference for homography calculations and court geometry understanding.

## Coordinate System and Origin

### **Origin and Coordinate System**
- **Origin (0,0)**: Located at the **top-left corner** of the reference image
- **X-axis**: Increases from **left to right** (horizontal)
- **Y-axis**: Increases from **top to bottom** (vertical)
- **Units**: Pixels in the reference coordinate space

### **Reference Image Dimensions**
```python
court_width = 1117 pixels          # Actual court width
court_height = 2408 pixels         # Actual court length  
top_bottom_border = 549 pixels     # Space above/below court
right_left_border = 274 pixels     # Space left/right of court

# Total reference image size
total_width = 1117 + 274*2 = 1665 pixels
total_height = 2408 + 549*2 = 3506 pixels
```

## Tennis Court Anatomy and Coordinates

### **Court Orientation**
The reference court is oriented **vertically** in the image:
- **Top baseline** (y=561): Where players serve from (top of image)
- **Bottom baseline** (y=2935): Where players serve from (bottom of image)
- **Net** (y=1748): Horizontal line across the middle
- **Length**: 2374 pixels (2935-561) representing 78 feet in real tennis
- **Width**: 1093 pixels (1379-286) representing 36 feet in real tennis

### **Key Court Lines and Coordinates**

#### **1. Baselines (Service Lines)**
```python
baseline_top = ((286, 561), (1379, 561))      # Top horizontal line
baseline_bottom = ((286, 2935), (1379, 2935)) # Bottom horizontal line
```

#### **2. Sidelines (Court Boundaries)**
```python
left_court_line = ((286, 561), (286, 2935))   # Left vertical boundary
right_court_line = ((1379, 561), (1379, 2935)) # Right vertical boundary
```

#### **3. Service Box Lines**
```python
left_inner_line = ((423, 561), (423, 2935))   # Left service box boundary
right_inner_line = ((1242, 561), (1242, 2935)) # Right service box boundary
top_inner_line = ((423, 1110), (1242, 1110))  # Top service line
bottom_inner_line = ((423, 2386), (1242, 2386)) # Bottom service line
```

#### **4. Center Lines**
```python
net = ((286, 1748), (1379, 1748))             # Net position
middle_line = ((832, 1110), (832, 2386))      # Center service line
```

#### **5. Extra Reference Points**
```python
top_extra_part = (832.5, 580)      # Center point above top baseline
bottom_extra_part = (832.5, 2910)  # Center point below bottom baseline
```

## 14 Key Points System

The system defines **14 key points** that correspond to important court intersections:

```python
key_points = [
    # Baseline corners (4 points)
    (286, 561),   # Top-left baseline
    (1379, 561),  # Top-right baseline  
    (286, 2935),  # Bottom-left baseline
    (1379, 2935), # Bottom-right baseline
    
    # Service box corners (4 points)
    (423, 561),   # Top-left service box
    (423, 2935),  # Bottom-left service box
    (1242, 561),  # Top-right service box
    (1242, 2935), # Bottom-right service box
    
    # Service line intersections (4 points)
    (423, 1110),  # Left top service line
    (1242, 1110), # Right top service line
    (423, 2386),  # Left bottom service line
    (1242, 2386), # Right bottom service line
    
    # Center line intersections (2 points)
    (832, 1110),  # Top center service
    (832, 2386)   # Bottom center service
]
```

## Real Tennis Court Mapping

### **Standard Tennis Court Dimensions**
- **Total Length**: 78 feet (23.77 meters)
- **Total Width**: 36 feet (10.97 meters)
- **Service Box Length**: 21 feet (6.40 meters)
- **Service Box Width**: 13.5 feet (4.11 meters)
- **Net Height**: 3 feet (0.91 meters) at center

### **Pixel-to-Real-World Scaling**
```python
# Court dimensions in pixels vs real world
court_length_pixels = 2374  # (2935 - 561)
court_width_pixels = 1093   # (1379 - 286)

# Scaling factors
length_scale = 2374 / 78 = 30.4 pixels per foot
width_scale = 1093 / 36 = 30.4 pixels per foot
```

## Court Configurations for Homography

The system defines **12 different court configurations**, each using 4 points to establish homography transformations:

### **Configuration Examples:**
1. **Full Court Rectangle** (Config 1): Uses all 4 baseline corners
2. **Service Box** (Config 2): Uses inner service box corners
3. **Left Half Court** (Config 3): Left inner + right outer boundaries
4. **Right Half Court** (Config 4): Left outer + right inner boundaries
5. **Service Area** (Config 5): Top and bottom service lines
6. **Top Service Box** (Config 6): Top service area
7. **Bottom Service Box** (Config 7): Bottom service area
8. **Right Service Area** (Config 8): Right side service lines
9. **Left Service Area** (Config 9): Left side service lines
10. **Left Center Service** (Config 10): Left side center service box
11. **Right Center Service** (Config 11): Right side center service box
12. **Bottom Service Line** (Config 12): Bottom service line area

## Usage in Homography Calculation

### **Process Flow:**
1. **Detection**: Model predicts 14 keypoints in broadcast video frame
2. **Configuration Testing**: Try each of the 12 configurations
3. **Homography Estimation**: For each config, use 4 detected + 4 reference points
4. **Validation**: Transform all reference points and measure error with remaining detections
5. **Selection**: Choose configuration with minimum transformation error

### **Benefits:**
- **Robustness**: Multiple configurations handle partial occlusions
- **Accuracy**: Uses geometric constraints of known court dimensions
- **Flexibility**: Works with different camera angles and court visibility

## Coordinate System Verification

To verify the coordinate system, key relationships should hold:

```python
# Court center should be at middle of width and length
center_x = (286 + 1379) / 2 = 832.5  ✓
center_y = (561 + 2935) / 2 = 1748    ✓ (matches net y-coordinate)

# Service box dimensions
service_box_width = 1242 - 423 = 819 pixels
service_box_length = 2386 - 1110 = 1276 pixels

# Net position (should be center of court length)
net_y = 1748
court_center_y = (561 + 2935) / 2 = 1748  ✓
```

## Practical Applications

### **1. Camera Calibration**
- Establish correspondence between broadcast camera view and reference court
- Calculate perspective transformation matrix

### **2. Player Tracking**
- Transform player positions from camera coordinates to court coordinates
- Measure distances and speeds in real-world units

### **3. Ball Tracking**
- Determine if ball is in/out of court boundaries
- Calculate ball trajectory in court coordinate system

### **4. Broadcast Graphics**
- Overlay graphics aligned with court geometry
- Add virtual elements (lines, statistics) in correct perspective

## Visual Representation

The `build_court_reference()` method creates a visual representation where:
- **White pixels (value=1)**: Court lines
- **Black pixels (value=0)**: Background
- **Line width**: 1 pixel (then dilated to 5x5 for visibility)

This generates `court_reference.png` showing the standardized court layout used throughout the system.

## Summary

The `CourtReference` class provides a **standardized, pixel-based coordinate system** for tennis court analysis with:

- **Origin**: Top-left corner (0,0)
- **Scale**: ~30.4 pixels per foot
- **Orientation**: Vertical court layout
- **Purpose**: Ground truth for homography and geometric corrections
- **Flexibility**: 12 configurations for robust homography estimation

This reference system enables accurate transformation between broadcast video coordinates and standardized court coordinates, supporting all downstream computer vision tasks in the tennis analysis pipeline.
