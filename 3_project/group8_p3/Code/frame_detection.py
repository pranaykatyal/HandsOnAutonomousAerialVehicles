import argparse
from pathlib import Path
import sys
import cv2 as cv
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Frame:
    """Represents a window frame with outer and inner contours"""
    outer_contour: np.ndarray
    inner_contours: List[np.ndarray]
    hierarchy_level: int
    area: float
    bounding_rect: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]

def find_frames(image: np.ndarray, min_area: int = 100) -> List[Frame]:
    """
    Detect rectangular frames in the image, handling occlusion using contour hierarchy.
    Assumes frames are white with darker backgrounds.
    """
    # Convert to grayscale if needed
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


    # Threshold to get binary image
    # For white frames on dark background
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    
    # Find contours with hierarchy
    contours, hierarchy = cv.findContours(
        binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    
    #if hierarchy is None or len(contours) == 0:
       # return []
    
    hierarchy = hierarchy[0]  # Unwrap hierarchy array
    frames = []
    processed_contours = set()  # Keep track of processed contours
    
    # First, find the largest frame (closest to screen)
    areas = [cv.contourArea(c) for c in contours]
    largest_idx = np.argmax(areas)
    
    def process_frame(idx: int, level: int = 0) -> Optional[Frame]:
        """Process a frame starting from its outer contour"""
        #if idx < 0 or idx in processed_contours:
            #return None
            
        contour = contours[idx]
        area = cv.contourArea(contour)
        
        # Skip if too small
        #if area < min_area:
           # return None
            
        # Check if roughly rectangular
        #peri = cv.arcLength(contour, True)
        #approx = cv.approxPolyDP(contour, 0.02 * peri, True)
        #if len(approx) < 4 or len(approx) > 8:  # Allow some deviation from perfect rectangle
         #   return None
            
        # Get bounding rectangle and center
        x, y, w, h = cv.boundingRect(contour)
        M = cv.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = x + w//2
            cy = y + h//2
            
        # Find inner contours (children in hierarchy)
        inner_contours = []
        child_idx = hierarchy[idx][2]  # First child
        while child_idx >= 0:
            child_contour = contours[child_idx]
            child_area = cv.contourArea(child_contour)
            
            # Check if child is a reasonable size compared to parent
            if child_area > 0.1 * area:  # Child should be at least 10% of parent
                peri = cv.arcLength(child_contour, True)
                approx = cv.approxPolyDP(child_contour, 0.02 * peri, True)
                # Check if child is roughly rectangular
                if 4 <= len(approx) <= 8:
                    inner_contours.append(child_contour)
            
            processed_contours.add(child_idx)
            child_idx = hierarchy[child_idx][0]  # Next sibling
            
        processed_contours.add(idx)
        
        return Frame(
            outer_contour=contour,
            inner_contours=inner_contours,
            hierarchy_level=level,
            area=area,
            bounding_rect=(x, y, w, h),
            center=(cx, cy)
        )
    
    # Process the largest frame first
    largest_frame = process_frame(largest_idx)
    if largest_frame:
        frames.append(largest_frame)
    
    # Then look for other frames at the same hierarchy level
    for i, (_, _, _, parent) in enumerate(hierarchy):
        if parent == hierarchy[largest_idx][3]:  # Same parent as largest frame
            if i != largest_idx and i not in processed_contours:
                frame = process_frame(i)
                if frame:
                    frames.append(frame)
    
    return frames

def draw_frame_analysis(image: np.ndarray, frames: List[Frame]) -> np.ndarray:
    """Draw detected frames with annotations - only shows largest frame"""
    result = image.copy()
    if len(image.shape) == 2:
        result = cv.cvtColor(result, cv.COLOR_GRAY2BGR)
    
    if not frames:
        return result
        
    # Find largest frame by area
    largest_frame = max(frames, key=lambda f: f.area)
    
    # Color scheme
    colors = [
        (0, 255, 0),    # Green for outer contour
        (0, 0, 255),    # Red for inner contours
        (255, 0, 0),    # Blue for centers
    ]
    
    # Draw outer contour
    cv.drawContours(result, [largest_frame.outer_contour], -1, colors[0], 2)
    
    # Draw inner contours
    cv.drawContours(result, largest_frame.inner_contours, -1, colors[1], 2)
    
    # Calculate and draw center using moments for better accuracy
    M = cv.moments(largest_frame.outer_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        x, y, w, h = largest_frame.bounding_rect
        cx = x + w//2
        cy = y + h//2
    
    # Draw center with larger, more visible markers
    cv.circle(result, (cx, cy), 8, (255, 255, 255), -1)  # white background
    cv.circle(result, (cx, cy), 6, colors[2], -1)  # blue center
    cv.circle(result, (cx, cy), 8, colors[2], 1)  # blue outline
    
    return result

def main():
    p = argparse.ArgumentParser(description="Detect window frames and their inner rectangles")
    p.add_argument("image", nargs="?", default="image.png",
                   help="Path to image (relative to script or absolute)")
    p.add_argument("--min-area", type=int, default=100,
                   help="Minimum contour area to consider")
    p.add_argument("--output", type=str, default="output.png",
                   help="Output path for annotated image")
    
    args = p.parse_args()
    
    # Resolve image path
    if Path(args.image).is_absolute():
        image_path = args.image
    else:
        image_path = str(Path(__file__).parent / args.image)
    
    # Load image
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'", file=sys.stderr)
        sys.exit(1)
    
    # Find frames
    frames = find_frames(image, args.min_area)
    
    if not frames:
        print("No frames detected!", file=sys.stderr)
        sys.exit(1)
    
    # Draw analysis
    result = draw_frame_analysis(image, frames)
    
    # Save result
    output_path = args.output
    if not Path(output_path).is_absolute():
        output_path = str(Path(__file__).parent / output_path)
        
    cv.imwrite(output_path, result)
    
    # Find and print info for largest frame only
    largest_frame = max(frames, key=lambda f: f.area)
    print("\nLargest Frame:")
    print(f"  Area: {largest_frame.area:.0f}")
    print(f"  Center: {largest_frame.center}")
    print(f"  Inner rectangles: {len(largest_frame.inner_contours)}")
    print(f"\nResult saved to {output_path}")

if __name__ == "__main__":
    main()
