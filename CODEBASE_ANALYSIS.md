# Tennis Court Detector - Comprehensive Codebase Analysis

## Project Overview

The Tennis Court Detector is a deep learning system designed to detect 14 key points of a tennis court from broadcast videos. The system uses a heatmap-based neural network architecture similar to TrackNet, enhanced with computer vision post-processing techniques for improved accuracy.

---

## 1. TRAINING PIPELINE

### 1.1 Main Training Loop (`main.py`)

**Architecture Setup:**
- **Model**: `BallTrackerNet` with 15 output channels (14 court keypoints + 1 center point)
- **Input Resolution**: 640×360 pixels  
- **Loss Function**: Mean Squared Error (MSE) between predicted and ground truth heatmaps
- **Optimizer**: Adam optimizer with learning rate 1e-5, β=(0.9, 0.999), no weight decay
- **Scheduler**: None explicitly defined

**Training Configuration:**
```python
# Default hyperparameters
batch_size = 2
num_epochs = 500
learning_rate = 1e-5
validation_intervals = 5 epochs
steps_per_epoch = 1000
```

**Training Process:**
1. **Data Loading**: Creates separate train/val datasets using `courtDataset` class
2. **Model Initialization**: Loads `BallTrackerNet` with custom weight initialization
3. **Logging Setup**: Uses TensorboardX for experiment tracking
4. **Training Loop**:
   - Runs training for specified steps per epoch
   - Validates every 5 epochs
   - Saves best model based on validation accuracy
   - Tracks metrics: loss, precision, accuracy, TP/FP/TN/FN

### 1.2 Training Function (`base_trainer.py`)

**Core Training Logic:**
```python
def train(model, train_loader, optimizer, criterion, device, epoch, max_iters=1000):
    model.train()
    for iter_id, batch in enumerate(train_loader):
        # Forward pass
        out = model(batch[0].float().to(device))
        gt_hm_hp = batch[1].float().to(device)
        
        # Apply sigmoid and compute loss
        loss = criterion(F.sigmoid(out), gt_hm_hp)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Key Features:**
- Applies sigmoid activation to model output before loss calculation
- Limited iterations per epoch (max_iters=1000) for faster training cycles
- Returns average loss per epoch

### 1.3 Neural Network Architecture (`tracknet.py`)

**BallTrackerNet Architecture:**
```
Input: (3, 360, 640) RGB image

Encoder (Downsampling):
Conv1-2 → MaxPool → Conv3-4 → MaxPool → Conv5-7 → MaxPool → Conv8-10
  64        ↓         128        ↓         256        ↓         512

Decoder (Upsampling):
Upsample → Conv11-13 → Upsample → Conv14-15 → Upsample → Conv16-18
   ↑          256         ↑          128         ↑          64→15

Output: (15, 360, 640) heatmaps
```

**ConvBlock Structure:**
- 3×3 Convolution → ReLU → BatchNorm
- All convolutions use padding=1, stride=1
- Custom weight initialization: uniform(-0.05, 0.05) for conv weights

**Output Channels:**
- Channels 0-13: Individual keypoint heatmaps
- Channel 14: Tennis court center point (for better convergence)

---

## 2. PREPROCESSING PIPELINE

### 2.1 Dataset Class (`dataset.py`)

**Data Organization:**
- **Dataset Path**: `./data/`
- **Images**: Stored in `./data/images/`
- **Annotations**: JSON files (`data_train.json`, `data_val.json`)
- **Original Resolution**: 1280×720 pixels
- **Network Resolution**: 640×360 pixels (scale factor = 2)

**Data Loading Process:**
```python
def __getitem__(self, index):
    # 1. Load image and resize
    img = cv2.imread(path)
    img = cv2.resize(img, (640, 360))
    
    # 2. Normalize to [0,1] and convert to CHW format
    inp = (img.astype(np.float32) / 255.)
    inp = np.rollaxis(inp, 2, 0)  # HWC → CHW
    
    # 3. Generate ground truth heatmaps
    hm_hp = np.zeros((15, 360, 640))
    for keypoint in keypoints:
        draw_gaussian(hm_hp[i], scaled_position, radius=55)
```

**Ground Truth Generation:**
- **Heatmap Size**: 15 channels × 360 × 640
- **Gaussian Radius**: 55 pixels
- **Center Point Calculation**: Intersection of court diagonal lines
- **Coordinate Scaling**: Original coordinates divided by 2 for network resolution

### 2.2 Gaussian Heatmap Generation (`utils.py`)

**Gaussian Function:**
```python
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return h
```

**Key Features:**
- **Diameter**: 2 × radius + 1 (111 pixels for radius=55)
- **Sigma**: diameter / 6 ≈ 18.5
- **Blending**: Uses `np.maximum()` to avoid overwriting existing peaks
- **Boundary Handling**: Clips gaussian to image boundaries

### 2.3 Data Filtering and Validation

**Filtering Criteria:**
- All keypoints must be within image boundaries (0 ≤ x ≤ 1280, 0 ≤ y ≤ 720)
- Removes samples with out-of-bounds annotations
- Ensures data quality for training

**Dataset Statistics:**
- **Total Images**: 8,841
- **Training Set**: 75% (~6,631 images)
- **Validation Set**: 25% (~2,210 images)
- **Court Types**: Hard, clay, grass courts from broadcast videos

---

## 3. POST-PROCESSING PIPELINE

### 3.1 Basic Heatmap Processing (`postprocess.py`)

**Primary Post-processing Function:**
```python
def postprocess(heatmap, scale=2, low_thresh=155, min_radius=10, max_radius=30):
    # 1. Threshold heatmap
    ret, heatmap = cv2.threshold(heatmap, 155, 255, cv2.THRESH_BINARY)
    
    # 2. Detect circles using Hough transform
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, 
                              dp=1, minDist=20, param1=50, param2=2,
                              minRadius=10, maxRadius=30)
    
    # 3. Scale coordinates back to original resolution
    if circles is not None:
        x_pred = circles[0][0][0] * 2  # scale=2
        y_pred = circles[0][0][1] * 2
    
    return x_pred, y_pred
```

**Process Flow:**
1. **Heatmap → Binary**: Threshold at intensity 155
2. **Circle Detection**: Hough circle transform to find peak regions
3. **Coordinate Scaling**: Multiply by scale factor (2) to return to original resolution
4. **Best Circle Selection**: Takes the first (strongest) detected circle

### 3.2 Keypoint Refinement (`postprocess.py`)

**Refinement Process:**
```python
def refine_kps(img, x_ct, y_ct, crop_size=40):
    # 1. Extract crop around predicted point
    img_crop = img[x_min:x_max, y_min:y_max]
    
    # 2. Detect lines in the crop
    lines = detect_lines(img_crop)
    
    # 3. Merge similar lines
    lines = merge_lines(lines)
    
    # 4. Find intersection of two main lines
    if len(lines) == 2:
        intersection = line_intersection(lines[0], lines[1])
        # Update coordinates if intersection is valid
```

**Line Detection Pipeline:**
- **Color Conversion**: BGR → Grayscale
- **Thresholding**: Binary threshold at 155
- **Line Detection**: Hough line transform with parameters:
  - `minLineLength=10`
  - `maxLineGap=30`
  - `threshold=30`

**Line Merging Logic:**
- **Distance Threshold**: 20 pixels for endpoint similarity
- **Merging Method**: Average of similar line endpoints
- **Goal**: Reduce multiple detections of same line to single representative

### 3.3 Homography Correction (`homography.py`)

**Court Reference System (`court_reference.py`):**
- **Reference Points**: 14 predefined court keypoints in ideal coordinates
- **Court Configurations**: 12 different 4-point combinations for homography estimation
- **Reference Dimensions**: 1117×2408 pixels (court) + borders

**Homography Estimation Process:**
```python
def get_trans_matrix(points):
    best_matrix = None
    min_error = infinity
    
    # Try all 12 court configurations
    for configuration in court_configurations:
        # Get 4 corresponding points
        detected_4pts = [points[i] for i in configuration_indices]
        reference_4pts = reference_configuration
        
        # Estimate homography matrix
        matrix = cv2.findHomography(reference_4pts, detected_4pts)
        
        # Transform all reference points
        transformed_points = cv2.perspectiveTransform(reference_points, matrix)
        
        # Calculate error with remaining detected points
        error = mean_distance(detected_points, transformed_points)
        
        # Keep best matrix
        if error < min_error:
            best_matrix = matrix
            min_error = error
```

**Correction Benefits:**
- **Occlusion Handling**: Uses geometric constraints to predict occluded points
- **Accuracy Improvement**: Leverages court geometry knowledge
- **Robustness**: Selects best configuration from 12 possible 4-point combinations

---

## 4. VALIDATION PIPELINE

### 4.1 Validation Function (`base_validator.py`)

**Evaluation Metrics:**
```python
def val(model, val_loader, criterion, device, epoch):
    model.eval()
    tp, fp, fn, tn = 0, 0, 0, 0
    max_dist = 7  # pixels
    
    for batch in val_loader:
        # Forward pass
        out = model(batch[0].float().to(device))
        pred = F.sigmoid(out).detach().cpu().numpy()
        
        # Process each keypoint
        for keypoint_num in range(14):
            heatmap = (pred[bs][keypoint_num] * 255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap)
            x_gt, y_gt = ground_truth[keypoint_num]
            
            # Calculate metrics
            if both_points_valid:
                distance = euclidean_distance((x_pred, y_pred), (x_gt, y_gt))
                if distance < 7: tp += 1
                else: fp += 1
            # Handle other cases (FN, TN)
```

**Performance Metrics:**
- **Distance Threshold**: 7 pixels for correct detection
- **Precision**: TP / (TP + FP)
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Confusion Matrix**: Tracks TP, FP, FN, TN for detailed analysis

### 4.2 Evaluation Results

**Baseline Performance:**
| Method | Precision | Accuracy | Median Distance |
|--------|-----------|----------|----------------|
| Base Model | 0.936 | 0.933 | 2.83 pixels |
| + Keypoint Refinement | 0.939 | 0.936 | 2.23 pixels |
| + Homography | 0.961 | 0.959 | 2.27 pixels |
| + Both Techniques | 0.963 | 0.961 | 1.83 pixels |

---

## 5. INFERENCE PIPELINE

### 5.1 Image Inference (`infer_in_image.py`)

**Inference Process:**
```python
# 1. Load and preprocess image
image = cv2.imread(input_path)
img = cv2.resize(image, (640, 360))
inp = normalize_and_tensorize(img)

# 2. Forward pass
out = model(inp.float().to(device))[0]
pred = F.sigmoid(out).detach().cpu().numpy()

# 3. Extract keypoints
points = []
for kps_num in range(14):
    heatmap = (pred[kps_num] * 255).astype(np.uint8)
    x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
    
    # Optional refinement
    if use_refine_kps and kps_num not in [8, 12, 9]:
        x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
    
    points.append((x_pred, y_pred))

# 4. Optional homography correction
if use_homography:
    matrix_trans = get_trans_matrix(points)
    if matrix_trans is not None:
        points = cv2.perspectiveTransform(refer_kps, matrix_trans)

# 5. Visualize results
for point in points:
    if point[0] is not None:
        cv2.circle(image, point, radius=0, color=(0,0,255), thickness=10)
```

### 5.2 Video Inference (`infer_in_video.py`)

**Video Processing Pipeline:**
```python
# 1. Read entire video into memory
frames, fps = read_video(input_path)

# 2. Process each frame
processed_frames = []
for frame in tqdm(frames):
    # Apply same inference pipeline as image
    keypoints = infer_keypoints(frame, model)
    annotated_frame = draw_keypoints(frame, keypoints)
    processed_frames.append(annotated_frame)

# 3. Write output video
write_video(processed_frames, fps, output_path)
```

**Key Differences from Image Inference:**
- **Batch Processing**: Processes all frames sequentially
- **Memory Management**: Loads entire video into memory (suitable for short clips)
- **Progress Tracking**: Uses tqdm for progress visualization
- **Output Format**: DIVX codec for video writing

---

## 6. KEY DESIGN DECISIONS AND INSIGHTS

### 6.1 Architecture Choices

**Why 15 Output Channels?**
- 14 channels for actual court keypoints
- 1 additional channel for court center point
- Center point helps with training convergence and stability

**Why 640×360 Resolution?**
- Balance between computational efficiency and detection accuracy
- 2× downscaling from original 1280×720 maintains aspect ratio
- Sufficient resolution for keypoint localization

### 6.2 Training Strategy

**Limited Steps Per Epoch:**
- 1000 steps per epoch rather than full dataset pass
- Enables faster experimentation and validation feedback
- Reduces overfitting risk with frequent validation

**Gaussian Heatmap Representation:**
- More robust than direct coordinate regression
- Handles uncertainty and multiple valid solutions
- Enables sub-pixel accuracy through peak localization

### 6.3 Post-processing Benefits

**Two-Stage Approach:**
1. **Neural Network**: Learns general keypoint locations from data
2. **Classical CV**: Refines predictions using domain knowledge (line intersections)

**Homography Integration:**
- Leverages known court geometry
- Provides correction for occluded or missed keypoints
- Improves overall system robustness

### 6.4 Evaluation Philosophy

**7-Pixel Threshold:**
- Represents practical accuracy for broadcast video analysis
- Accounts for annotation uncertainty and camera motion
- Balances strictness with real-world applicability

---

## 7. POTENTIAL IMPROVEMENTS

### 7.1 Architecture Enhancements
- **Multi-scale Features**: Add skip connections or feature pyramid networks
- **Attention Mechanisms**: Focus on court line regions
- **Temporal Consistency**: For video sequences, add temporal modeling

### 7.2 Training Improvements
- **Data Augmentation**: Geometric transforms, color jittering, synthetic blur
- **Loss Function**: Focal loss for hard examples, geometric consistency losses
- **Learning Rate Scheduling**: Cosine annealing or step decay

### 7.3 Post-processing Refinements
- **Robust Line Fitting**: RANSAC for line detection in noisy conditions
- **Multi-frame Tracking**: Temporal smoothing for video sequences
- **Confidence Estimation**: Uncertainty quantification for predictions

This comprehensive analysis provides a complete understanding of the Tennis Court Detector system, from data preprocessing through training to inference and evaluation.
