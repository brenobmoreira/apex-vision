import cv2
import numpy as np
import os
from ultralytics import YOLO

def fit_line_and_get_angle(mask_bool, image_to_draw_on):
    """
    Ajusta uma linha à máscara, desenha-a com comprimento fixo e retorna o ângulo.
    """
    points = np.column_stack(np.where(mask_bool))
    
    if len(points) < 5:
        return None

    # Converte os pontos para o formato (x, y) e o tipo necessários
    points_xy = points[:, ::-1].astype(np.float32)
    
    # Ajusta a linha
    line = cv2.fitLine(points_xy, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()
    
    # --- ALTERAÇÃO 1: Linha com comprimento fixo de 150 pixels ---
    line_length = 150
    p1_x = int(x0 - line_length / 2 * vx)
    p1_y = int(y0 - line_length / 2 * vy)
    p2_x = int(x0 + line_length / 2 * vx)
    p2_y = int(y0 + line_length / 2 * vy)
    
    # Desenha a linha na imagem
    cv2.line(image_to_draw_on, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 2)
    
    # Calcula e retorna o ângulo em graus
    angle_rad = np.arctan2(vy, vx)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def draw_text_with_shadow(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, color=(255, 255, 255), thickness=2):
    """Desenha texto com uma sombra preta para melhor legibilidade."""
    cv2.putText(image, text, position, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA) # Sombra mais espessa
    cv2.putText(image, text, position, font, scale, color, thickness, cv2.LINE_AA)

def display_image_segmentation(image_path, model, confidence=0.2):
    class_names = model.names
    print(f"Classes: {list(class_names.values())}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    print(f"Running inference on {image_path} with confidence: {confidence}")
    results = model(image, conf=confidence, verbose=False)
    
    result_image = image.copy()
    
    if results[0].masks is None:
        print("No detections found")
        return

    masks_data = []
    # --- PRIMEIRA PASSAGEM: Coletar todas as máscaras e ângulos absolutos ---
    for i, mask_tensor in enumerate(results[0].masks.data):
        confidence = float(results[0].boxes.conf[i].cpu().numpy())

        # Pula detecções com confiança baixa
        if confidence <= 0.75:
            continue
            
        mask_np = mask_tensor.cpu().numpy()
        class_id = int(results[0].boxes.cls[i].cpu().numpy())
        class_name = class_names.get(class_id, "Unknown")
        
        h, w = image.shape[:2]
        mask_resized = cv2.resize(mask_np, (w, h))
        mask_bool = mask_resized > 0.5
        
        absolute_angle = fit_line_and_get_angle(mask_bool, result_image)
        
        if absolute_angle is not None:
            y, x = np.where(mask_bool)
            center_x = int(np.mean(x))
            center_y = int(np.mean(y))
            
            masks_data.append({
                "id": i,
                "class_name": class_name,
                "center": (center_x, center_y),
                "confidence": confidence,
                "absolute_angle": absolute_angle
            })
            draw_text_with_shadow(result_image, str(i), (center_x, center_y), scale=0.9, thickness=3)

    angle_ref = None
    for data in masks_data:
        if data["class_name"] == 'L1':
            angle_ref = data["absolute_angle"]
            break
            
    if angle_ref is None:
        print("Warning: Reference class 'L1' not found. Cannot calculate relative angles.")

    # --- SEGUNDA PASSAGEM: Calcular ângulos relativos e exibir ---
    y_offset = 35
    # --- ALTERAÇÃO 2: Título dos ângulos com fonte maior ---
    draw_text_with_shadow(result_image, "Angulos:", (10, y_offset), scale=0.8, thickness=2)
    
    masks_data.sort(key=lambda x: x['id'])

    for data in masks_data:
        relative_angle = 0.0
        if angle_ref is not None:
            if data["class_name"] == 'L1':
                relative_angle = 0.0
            else:
                diff = data["absolute_angle"] - angle_ref
                relative_angle = (diff + 180) % 360 - 180
                relative_angle = abs(relative_angle)
                if relative_angle > 90:
                    relative_angle = 180 - relative_angle

        # Exibe ID, ângulo e confiança
        y_offset += 25
        angle_text = f"ID {data['id']}: {relative_angle:.1f}°"
        conf_text = f"  (conf: {data['confidence']:.2f})"
        
        # Desenha o ID e ângulo em branco com sombra preta
        draw_text_with_shadow(result_image, angle_text, (10, y_offset), scale=0.7, thickness=2)
        
        # Desenha a confiança em cinza claro
        text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(result_image, conf_text, (15 + text_size[0], y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        print(f"ID {data['id']} ({data['class_name']}): Conf={data['confidence']:.2f}, Abs Angle={data['absolute_angle']:.2f}, Rel Angle={relative_angle:.2f}")

    cv2.imshow('Detection Results', result_image)
    print("\nPressione qualquer tecla para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    test_files = ["7"]
    for test_id in test_files:
        input_image = f'extracted_frames/teste{test_id}.png'
        if not os.path.exists(input_image):
            print(f"Error: Image file not found at {input_image}")
            continue
        
        display_image_segmentation(input_image, model, confidence=0.2)

if __name__ == "__main__":
    main()