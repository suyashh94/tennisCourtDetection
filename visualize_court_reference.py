#!/usr/bin/env python3
"""
Visualization script to understand the court reference coordinate system.
This script creates annotated visualizations showing the coordinate system, 
key points, and court configurations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from court_reference import CourtReference

def visualize_court_with_coordinates():
    """Create a detailed visualization of the court reference system."""
    
    # Create court reference
    court_ref = CourtReference()
    court_img = court_ref.build_court_reference()
    
    # Convert to color for annotations
    court_color = cv2.cvtColor(court_img * 255, cv2.COLOR_GRAY2BGR)
    
    # Define colors
    COLORS = {
        'baseline': (0, 255, 0),      # Green
        'sideline': (255, 0, 0),      # Blue  
        'service': (0, 0, 255),       # Red
        'net': (255, 255, 0),         # Cyan
        'center': (255, 0, 255),      # Magenta
        'keypoint': (0, 255, 255),    # Yellow
        'text': (255, 255, 255)       # White
    }
    
    # Draw and label key lines with different colors
    lines_info = [
        (court_ref.baseline_top, 'baseline', 'Top Baseline'),
        (court_ref.baseline_bottom, 'baseline', 'Bottom Baseline'),
        (court_ref.left_court_line, 'sideline', 'Left Sideline'),
        (court_ref.right_court_line, 'sideline', 'Right Sideline'),
        (court_ref.left_inner_line, 'service', 'Left Service'),
        (court_ref.right_inner_line, 'service', 'Right Service'),
        (court_ref.top_inner_line, 'service', 'Top Service'),
        (court_ref.bottom_inner_line, 'service', 'Bottom Service'),
        (court_ref.net, 'net', 'Net'),
        (court_ref.middle_line, 'center', 'Center Line')
    ]
    
    # Draw lines with colors
    for line_coords, color_key, label in lines_info:
        color = COLORS[color_key]
        cv2.line(court_color, line_coords[0], line_coords[1], color, 3)
    
    # Draw and number the 14 key points
    key_points = court_ref.key_points
    point_labels = [
        'BL-TL', 'BL-TR', 'BL-BL', 'BL-BR',  # Baseline corners
        'SB-TL', 'SB-BL', 'SB-TR', 'SB-BR',  # Service box corners  
        'SL-TL', 'SL-TR', 'SL-BL', 'SL-BR',  # Service line intersections
        'CT-T', 'CT-B'                        # Center line intersections
    ]
    
    for i, (point, label) in enumerate(zip(key_points, point_labels)):
        # Draw circle for keypoint
        cv2.circle(court_color, point, 8, COLORS['keypoint'], -1)
        cv2.circle(court_color, point, 8, (0, 0, 0), 2)
        
        # Add point number
        cv2.putText(court_color, str(i+1), 
                   (point[0]-10, point[1]-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 2)
        
        # Add point label
        cv2.putText(court_color, label, 
                   (point[0]-20, point[1]+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1)
    
    # Add coordinate system annotations
    # Origin marker
    cv2.circle(court_color, (0, 0), 15, (255, 255, 255), 3)
    cv2.putText(court_color, 'ORIGIN (0,0)', (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['text'], 2)
    
    # Axis arrows and labels
    cv2.arrowedLine(court_color, (50, 80), (150, 80), (255, 255, 255), 3)
    cv2.putText(court_color, 'X-axis', (160, 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 2)
    
    cv2.arrowedLine(court_color, (50, 80), (50, 180), (255, 255, 255), 3)
    cv2.putText(court_color, 'Y-axis', (10, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 2)
    
    # Add dimensions
    cv2.putText(court_color, f'Court: {court_ref.court_width}x{court_ref.court_height}px', 
               (50, court_color.shape[0]-100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['text'], 2)
    
    cv2.putText(court_color, f'Total: {court_ref.court_total_width}x{court_ref.court_total_height}px', 
               (50, court_color.shape[0]-70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['text'], 2)
    
    # Save the annotated court
    cv2.imwrite('court_reference_annotated.png', court_color)
    print("Created court_reference_annotated.png")
    
    return court_color

def create_coordinate_system_diagram():
    """Create a simple diagram showing the coordinate system."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Left plot: Coordinate system explanation
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    
    # Draw coordinate system
    ax1.arrow(1, 1, 3, 0, head_width=0.2, head_length=0.2, fc='red', ec='red')
    ax1.arrow(1, 1, 0, 3, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    
    ax1.text(4.5, 0.5, 'X-axis (Left→Right)', fontsize=12, color='red')
    ax1.text(0.2, 4.5, 'Y-axis\n(Top→Bottom)', fontsize=12, color='blue', rotation=90)
    ax1.text(0.5, 0.5, 'Origin\n(0,0)', fontsize=10, ha='center')
    
    ax1.set_title('Reference Coordinate System', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Right plot: Court layout with key coordinates
    court_ref = CourtReference()
    
    # Simplified court representation
    ax2.set_xlim(200, 1450)
    ax2.set_ylim(3000, 500)  # Inverted Y for image coordinates
    ax2.set_aspect('equal')
    
    # Draw court outline
    court_coords = [
        [286, 1379, 1379, 286, 286],  # X coordinates
        [561, 561, 2935, 2935, 561]   # Y coordinates  
    ]
    ax2.plot(court_coords[0], court_coords[1], 'k-', linewidth=3, label='Court Boundary')
    
    # Draw net
    ax2.plot([286, 1379], [1748, 1748], 'r-', linewidth=2, label='Net')
    
    # Draw service lines
    ax2.plot([423, 1242], [1110, 1110], 'b-', linewidth=1, label='Service Lines')
    ax2.plot([423, 1242], [2386, 2386], 'b-', linewidth=1)
    ax2.plot([423, 423], [1110, 2386], 'b-', linewidth=1)
    ax2.plot([1242, 1242], [1110, 2386], 'b-', linewidth=1)
    ax2.plot([832, 832], [1110, 2386], 'g-', linewidth=1, label='Center Line')
    
    # Add coordinate annotations
    key_coords = [
        ((286, 561), 'TL:(286,561)'),
        ((1379, 561), 'TR:(1379,561)'),
        ((286, 2935), 'BL:(286,2935)'),
        ((1379, 2935), 'BR:(1379,2935)'),
        ((832, 1748), 'Net:(832,1748)')
    ]
    
    for (x, y), label in key_coords:
        ax2.plot(x, y, 'ro', markersize=8)
        ax2.annotate(label, (x, y), xytext=(10, 10), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
    
    ax2.set_title('Tennis Court Reference Layout', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X Coordinate (pixels)')
    ax2.set_ylabel('Y Coordinate (pixels)')
    
    plt.tight_layout()
    plt.savefig('coordinate_system_explanation.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Created coordinate_system_explanation.png")

def show_court_configurations():
    """Visualize the 12 different court configurations used for homography."""
    
    court_ref = CourtReference()
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (conf_id, points) in enumerate(court_ref.court_conf.items()):
        ax = axes[i]
        
        # Draw simplified court
        ax.set_xlim(200, 1450)
        ax.set_ylim(3000, 500)
        ax.set_aspect('equal')
        
        # Court outline
        court_x = [286, 1379, 1379, 286, 286]
        court_y = [561, 561, 2935, 2935, 561]
        ax.plot(court_x, court_y, 'k-', linewidth=1, alpha=0.5)
        
        # Net
        ax.plot([286, 1379], [1748, 1748], 'r-', linewidth=1, alpha=0.5)
        
        # Service lines
        ax.plot([423, 1242, 1242, 423, 423], [1110, 1110, 2386, 2386, 1110], 'b-', linewidth=1, alpha=0.5)
        ax.plot([832, 832], [1110, 2386], 'g-', linewidth=1, alpha=0.5)
        
        # Highlight the 4 points for this configuration
        conf_x = [p[0] for p in points]
        conf_y = [p[1] for p in points]
        
        # Draw the configuration rectangle/polygon
        if len(points) == 4:
            # Connect the 4 points to form a quadrilateral
            ax.plot([conf_x[0], conf_x[1], conf_x[3], conf_x[2], conf_x[0]], 
                   [conf_y[0], conf_y[1], conf_y[3], conf_y[2], conf_y[0]], 
                   'ro-', linewidth=3, markersize=8, alpha=0.8)
        
        ax.set_title(f'Config {conf_id}', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('12 Court Configurations for Homography Estimation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('court_configurations.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Created court_configurations.png")

if __name__ == '__main__':
    print("Creating court reference visualizations...")
    
    # Create annotated court image
    visualize_court_with_coordinates()
    
    # Create coordinate system explanation
    create_coordinate_system_diagram()
    
    # Show court configurations
    show_court_configurations()
    
    print("\nVisualization files created:")
    print("1. court_reference_annotated.png - Detailed court with coordinates")
    print("2. coordinate_system_explanation.png - Coordinate system diagram") 
    print("3. court_configurations.png - 12 homography configurations")
    print("4. court_reference.png - Basic court reference (from court_reference.py)")
