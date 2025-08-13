import cv2
import numpy as np
import os
from ultralytics import YOLO

def fit_line_and_get_angle(mask_bool, image_to_draw_on):
    """
    Ajusta uma linha à máscara usando cv2.fitLine, desenha-a e retorna o ângulo.
    """
    points = np.column_stack(np.where(mask_bool))
    
    if len(points) < 5:
        return None  # Não há pontos suficientes

    # Converte os pontos para o formato (x, y) e o tipo necessários
    points = points[:, ::-1].astype(np.float32)
    
    # Ajusta a linha
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()
    
    # Define um comprimento fixo para a linha e calcula seus pontos
    line_length = 150
    p1_x = int(x0 - line_length / 2 * vx)
    p1_y = int(y0 - line_length / 2 * vy)
    p2_x = int(x0 + line_length / 2 * vx)
    p2_y = int(y0 + line_length / 2 * vy)
    
    # Desenha a linha na imagem
    cv2.line(image_to_draw_on, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 2)  # Linha vermelha
    
    # Calcula e retorna o ângulo em graus
    angle_rad = np.arctan2(vy, vx)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def min_area_rect_and_get_angle(mask_bool, image_to_draw_on):
    """
    Encontra o retângulo de área mínima, desenha-o e retorna o ângulo.
    """
    points = np.column_stack(np.where(mask_bool))
    
    if len(points) < 5:
        return None

    points = points[:, ::-1]  # Converte para (x, y)
    
    rect = cv2.minAreaRect(points)
    angle = rect[2]
    
    # Corrige o ângulo retornado pela função
    if rect[1][0] < rect[1][1]:
        angle += 90
        
    # Desenha a caixa do retângulo para visualização
    box = np.int64(cv2.boxPoints(rect))
    cv2.drawContours(image_to_draw_on, [box], 0, (0, 255, 255), 2)  # Caixa amarela

    return angle

def draw_square(result_image):
    h_final, w_final = result_image.shape[:2]
    square_size_final = 150
    top_left = (w_final // 2 - square_size_final // 2, h_final // 2 - square_size_final // 2)
    bottom_right = (w_final // 2 + square_size_final // 2, h_final // 2 + square_size_final // 2)
    cv2.rectangle(result_image, top_left, bottom_right, (0, 0, 0), 2)

def display_image_segmentation(image_path, model_path, confidence=0.2):
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    class_names = model.names
    print(f"Classes: {list(class_names.values())}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    print(f"Running inference with confidence: {confidence}")
    results = model(image, conf=confidence, verbose=False)
    
    result_image = image.copy()
    detection_count = 0
    
    # Dicionário para armazenar os ângulos de cada máscara
    angles_fitline = {}
    angle_0 = None  # Inicializa a variável para o ângulo de referência

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes
        classes = boxes.cls.cpu().numpy()
        
        detection_count = len(masks)
        print(f"Found {detection_count} detections")
        
        for mask_id, (mask, cls_id) in enumerate(zip(masks, classes)):
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

            color = (0, 255, 0) if cls_id == 0 else (255, 0, 0) # Verde para ID 0, Azul para ID 1
            
            # Aplica a cor na máscara
            blended = cv2.addWeighted(
                result_image[mask_bool], 0.7,
                np.full_like(result_image[mask_bool], color), 0.3, 0
            )
            # Obtém o ângulo absoluto da linha
            angle1 = fit_line_and_get_angle(mask_bool, result_image)
            # Verifica se esta é a classe L1 (usando class_names para mapear o ID para o nome da classe)
            class_name = class_names.get(int(cls_id), str(cls_id))
            if class_name == 'L1' and angle1 is not None:
                angle_0 = angle1
                angles_fitline[int(mask_id)] = 0
                print(f"Ângulo de referência (L1) definido como: {angle_0:.2f}°")

            
            
            # Se não for L1 e angle_0 já estiver definido, calcula o ângulo relativo
            if class_name != 'L1' and angle_0 is not None and angle1 is not None:
                # Calcula a diferença angular, garantindo que fique entre -180 e 180 graus
                relative_angle = (angle1 - angle_0 + 180) % 360 - 180
                # Se o ângulo for negativo, pega o complemento positivo
                if relative_angle < 0:
                    relative_angle = 360 + relative_angle
                angles_fitline[int(mask_id)] = relative_angle
                angle1 = relative_angle  # Atualiza o ângulo para exibição
                print(f"Ângulo relativo para {class_name}: {relative_angle:.2f}° (referência: {angle_0:.2f}°)")
            
            # Encontra o centro da máscara para posicionar o texto
            y, x = np.where(mask_bool)
            if len(x) > 0 and len(y) > 0:
                center_x = int(np.mean(x))
                center_y = int(np.mean(y))
                
                # Desenha apenas o ID ao lado da linha
                id_text = f"{mask_id}"
                text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Desenha fundo preto para o ID
                cv2.rectangle(
                    result_image,
                    (center_x - 5, center_y - text_size[1] - 5),
                    (center_x + text_size[0] + 5, center_y + 5),
                    (0, 0, 0), -1
                )
                
                # Desenha o ID
                cv2.putText(
                    result_image, id_text, (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                    
        # Exibe os ângulos no canto superior esquerdo
        y_offset = 30
        cv2.putText(result_image, "Ângulos:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(result_image, "Ângulos:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for idx, (mask_id, angle) in enumerate(angles_fitline.items()):
            y_offset += 25
            angle_text = f"ID {mask_id}: {angle:.1f}°"
            cv2.putText(result_image, angle_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(result_image, angle_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        print("Ângulos calculados:", angles_fitline)
    else:
        print("No detections found")

    cv2.imshow('Detection Results', result_image)
    print("\nPressione qualquer tecla para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return True

def main():
    
    test = ["6"]
    for i in range(len(test)):
        input_image = f'extracted_frames/teste{test[i]}.png' # Coloque o caminho da sua imagem
        model = 'best.pt' # Coloque o caminho do seu modelo

        display_image_segmentation(input_image, model, confidence=0.2)

if __name__ == "__main__":
    main()