import cv2
import numpy as np
import argparse
import os
from ultralytics import YOLO


def display_image_segmentation(image_path, model_path, confidence=0.2):
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    class_names = model.names
    print(f"Classes: {list(class_names.values())}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    print(f"Image shape: {image.shape}")
    
    print(f"Running inference with confidence: {confidence}")
    results = model(image, conf=confidence, verbose=False)
    
    result_image = image.copy()
    detection_count = 0
    
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes
        classes = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        
        detection_count = len(masks)
        print(f"Found {detection_count} detections")
        
        for mask_id, (mask, cls_id, conf_score) in enumerate(zip(masks, classes, confidences)):
            # Agora cada máscara tem um id: mask_id
            # Exemplo de uso:
            # print(f"Máscara ID: {mask_id}, Classe: {class_names[int(cls_id)]}, Score: {conf_score:.3f}")
            
            h, w = image.shape[:2]
            mask_resized = cv2.resize(mask, (w, h))
            mask_bool = mask_resized > 0.5

            square_size = 150
            h_img, w_img = result_image.shape[:2]
            x0 = w_img // 2 - square_size // 2
            y0 = h_img // 2 - square_size // 2
            x1 = x0 + square_size
            y1 = y0 + square_size
            mask_inside_square = np.zeros_like(mask_bool)
            mask_inside_square[y0:y1, x0:x1] = mask_bool[y0:y1, x0:x1]

            color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)  # Green/Blue
            if np.any(mask_inside_square):
                blended = cv2.addWeighted(
                    result_image[mask_inside_square], 0.7,
                    np.full_like(result_image[mask_inside_square], color), 0.3, 0
                )
                result_image[mask_inside_square] = blended

                # Encontrar contornos da máscara dentro do quadrado
                mask_uint8 = (mask_inside_square * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    cnt = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = box.astype(int)

                    # Desenhar círculos nas quinas
                    for pt in box:
                        cv2.circle(result_image, tuple(pt), 2, (0,0,255), -1)

                    # Garantir que box está em formato numpy array (4,2)
                    box = np.array(box)
                    # Calcular os comprimentos dos lados
                    side_lengths = [np.linalg.norm(box[i] - box[(i+1)%4]) for i in range(4)]
                    # Encontrar o maior lado
                    max_idx = np.argmax(side_lengths)
                    # Os lados opostos são (max_idx, max_idx+1) e (max_idx+2, max_idx+3)
                    i1, i2 = max_idx, (max_idx+1)%4
                    i3, i4 = (max_idx+2)%4, (max_idx+3)%4
                    # Pontos médios dos lados mais longos
                    mid1 = ((box[i1][0]+box[i2][0])//2, (box[i1][1]+box[i2][1])//2)
                    mid2 = ((box[i3][0]+box[i4][0])//2, (box[i3][1]+box[i4][1])//2)
                    # Traçar linha central (vermelha, fina)
                    cv2.line(result_image, mid1, mid2, (0,0,255), 1)


    else:
        print("No detections found")
        cv2.putText(result_image, "No detections found", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    h, w = result_image.shape[:2]
    square_size = 150
    top_left = (w // 2 - square_size // 2, h // 2 - square_size // 2)
    bottom_right = (w // 2 + square_size // 2, h // 2 + square_size // 2)
    cv2.rectangle(result_image, top_left, bottom_right, (0, 0, 0), 2)

    cv2.imshow('Detection Results', result_image)
    print(f"\nDisplaying image with {detection_count} detections")
    print("Press any key to close window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return True


def main():
    # parser = argparse.ArgumentParser(description='Display YOLO Image Results')
    # parser.add_argument('input_image', help='Path to input image')
    # parser.add_argument('-m', '--model', required=True, help='Path to model weights')
    # parser.add_argument('-c', '--confidence', type=float, default=0.2, help='Confidence threshold')
    
    # args = parser.parse_args()
    
    # if not os.path.exists(args.input_image):
    #     print(f"Error: Image '{args.input_image}' not found")
    #     return
    
    # if not os.path.exists(args.model):
    #     print(f"Error: Model '{args.model}' not found")
    #     return
    
    input_image = 'extracted_frames/frame_000711.jpg'
    model = 'best.pt'
    confidence = 0.2

    display_image_segmentation(input_image, model, confidence)


if __name__ == "__main__":
    main()