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

    points_xy = points[:, ::-1].astype(np.float32)
    
    line = cv2.fitLine(points_xy, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()
    
    line_length = 150
    p1_x = int(x0 - line_length / 2 * vx)
    p1_y = int(y0 - line_length / 2 * vy)
    p2_x = int(x0 + line_length / 2 * vx)
    p2_y = int(y0 + line_length / 2 * vy)
    
    cv2.line(image_to_draw_on, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 2)
    
    angle_rad = np.arctan2(vy, vx)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def draw_text_with_shadow(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, color=(255, 255, 255), thickness=2):
    """Desenha texto com uma sombra preta para melhor legibilidade."""
    cv2.putText(image, text, position, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, position, font, scale, color, thickness, cv2.LINE_AA)

def process_frame(frame, model, angle_ref=None):
    """Processa um único frame do vídeo e retorna o frame processado e o ângulo de referência."""
    if frame is None:
        return None, angle_ref
    
    result_image = frame.copy()
    results = model(frame, conf=0.2, verbose=False)
    
    if results[0].masks is None:
        return result_image, angle_ref
    
    masks_data = []
    # Primeira passagem: Coletar todas as máscaras e ângulos absolutos
    for i, mask_tensor in enumerate(results[0].masks.data):
        mask_np = mask_tensor.cpu().numpy()
        class_id = int(results[0].boxes.cls[i].cpu().numpy())
        class_name = model.names.get(class_id, "Unknown")
        
        h, w = frame.shape[:2]
        mask_resized = cv2.resize(mask_np, (w, h))
        mask_bool = mask_resized > 0.5
        
        absolute_angle = fit_line_and_get_angle(mask_bool, result_image)
        
        if absolute_angle is not None:
            y, x = np.where(mask_bool)
            center_x = int(np.mean(x))
            center_y = int(np.mean(y))
            
            confidence = float(results[0].boxes.conf[i].cpu().numpy())
            
            masks_data.append({
                "id": i,
                "class_name": class_name,
                "absolute_angle": absolute_angle,
                "center": (center_x, center_y),
                "confidence": confidence,
            })
            draw_text_with_shadow(result_image, str(i), (center_x, center_y), scale=0.9, thickness=3)
    
    # Atualiza o ângulo de referência se encontrar um objeto L1
    for data in masks_data:
        if data["class_name"] == 'L1':
            angle_ref = data["absolute_angle"]
            break
    
    # Segunda passagem: Calcular ângulos relativos
    y_offset = 35
    draw_text_with_shadow(result_image, "Angulos:", (10, y_offset), scale=0.8, thickness=2)
    
    for data in sorted(masks_data, key=lambda x: x['id']):
        relative_angle = 0.0
        if angle_ref is not None and data["class_name"] != 'L1':
            diff = data["absolute_angle"] - angle_ref
            relative_angle = (diff + 180) % 360 - 180
            relative_angle = abs(relative_angle)
            if relative_angle > 90:
                relative_angle = 180 - relative_angle
        
        y_offset += 25
        angle_text = f"ID {data['id']}: {relative_angle:.1f}°"
        conf_text = f"  (conf: {data['confidence']:.2f})"
        
        draw_text_with_shadow(result_image, angle_text, (10, y_offset), scale=0.7, thickness=2)
        text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(result_image, conf_text, (15 + text_size[0], y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    
    return result_image, angle_ref

def process_video(input_video_path, output_video_path, model):
    """Processa um vídeo e salva o resultado."""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {input_video_path}")
        return
    
    # Configuração do vídeo de saída
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    angle_ref = None
    frame_count = 0
    
    print(f"Processando vídeo: {input_video_path}")
    print("Pressione 'q' para sair ou 'p' para pausar")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Processa o frame
        processed_frame, angle_ref = process_frame(frame, model, angle_ref)
        
        if processed_frame is None:
            break
            
        # Mostra o frame processado
        cv2.imshow('Video Analysis', processed_frame)
        out.write(processed_frame)
        
        frame_count += 1
        print(f"Frame processado: {frame_count}", end='\r')
        
        # Controles do teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Sair
            break
        elif key == ord('p'):  # Pausar
            while True:
                key = cv2.waitKey(1)
                if key == ord('p') or key == ord('q'):
                    break
            if key == ord('q'):
                break
    
    # Limpeza
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\nProcessamento concluído!")

def main():
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        print(f"Erro: Arquivo do modelo não encontrado em {model_path}")
        return
    
    print(f"Carregando modelo: {model_path}")
    model = YOLO(model_path)
    
    # Exemplo de uso
    input_video = 'videos/video_new.mp4'  # Substitua pelo caminho do seu vídeo
    output_video = 'videos/resultado_analise.mp4'
    
    if not os.path.exists(input_video):
        print(f"Erro: Arquivo de vídeo não encontrado em {input_video}")
        return
    
    process_video(input_video, output_video, model)

if __name__ == "__main__":
    main()
