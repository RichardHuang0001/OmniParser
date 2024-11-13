from typing import Optional
import gradio as gr
import numpy as np
import torch
from PIL import Image
import io
import json
import base64, os
from utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
import torch
from PIL import Image

yolo_model = get_yolo_model(model_path='weights/icon_detect/best.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")

MARKDOWN = """
# OmniParser for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""

DEVICE = torch.device('cuda')

def process(
    image_input,
    box_threshold,
    iou_threshold,
    use_paddleocr,
    imgsz
) -> Optional[tuple]:
    """
    处理图像并返回检测结果
    Returns:
        tuple: (标注后的图像, 解析的内容列表, bbox坐标信息)
    """
    image_save_path = 'imgs/saved_image_demo.png'
    image_input.save(image_save_path)
    image = Image.open(image_save_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    # 获取OCR结果
    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_save_path, 
        display_img=False, 
        output_bb_format='xyxy', 
        goal_filtering=None, 
        easyocr_args={'paragraph': False, 'text_threshold':0.9}, 
        use_paddleocr=use_paddleocr
    )
    text, ocr_bbox = ocr_bbox_rslt

    # 获取检测结果
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_save_path, 
        yolo_model, 
        BOX_TRESHOLD=box_threshold, 
        output_coord_in_ratio=True, 
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config, 
        caption_model_processor=caption_model_processor, 
        ocr_text=text,
        iou_threshold=iou_threshold, 
        imgsz=imgsz
    )

    # 处理输出图像
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    
    # 处理文本输出
    parsed_content_text = '\n'.join(parsed_content_list)
    
    # 处理bbox坐标信息
    bbox_info = {}
    for idx, (element_id, coords) in enumerate(label_coordinates.items()):
        # 将相对坐标转换为绝对坐标
        abs_coords = [
            coords[0] * image.size[0],  # x1
            coords[1] * image.size[1],  # y1
            coords[2] * image.size[0],  # x2
            coords[3] * image.size[1]   # y2
        ]
        
        element_type = "text" if idx < len(text) else "icon"
        bbox_info[element_id] = {
            "type": element_type,
            "coordinates": {
                "x1": round(abs_coords[0], 2),
                "y1": round(abs_coords[1], 2),
                "x2": round(abs_coords[2], 2),
                "y2": round(abs_coords[3], 2)
            },
            "description": parsed_content_list[idx] if idx < len(parsed_content_list) else ""
        }
    
    bbox_json = json.dumps(bbox_info, indent=2, ensure_ascii=False)
    
    return image, parsed_content_text, bbox_json

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image_input_component = gr.Image(type='pil', label='Upload image')
            box_threshold_component = gr.Slider(
                label='Box Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.05)
            iou_threshold_component = gr.Slider(
                label='IOU Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.1)
            use_paddleocr_component = gr.Checkbox(
                label='Use PaddleOCR', value=True)
            imgsz_component = gr.Slider(
                label='Icon Detect Image Size', minimum=640, maximum=1920, step=32, value=640)
            submit_button_component = gr.Button(
                value='Submit', variant='primary')
        
        with gr.Column():
            image_output_component = gr.Image(type='pil', label='Image Output')
            text_output_component = gr.Textbox(
                label='Parsed screen elements', 
                placeholder='Text Output'
            )
            bbox_output_component = gr.JSON(
                label='Bounding Box Coordinates'
            )

    submit_button_component.click(
        fn=process,
        inputs=[
            image_input_component,
            box_threshold_component,
            iou_threshold_component,
            use_paddleocr_component,
            imgsz_component
        ],
        outputs=[
            image_output_component, 
            text_output_component,
            bbox_output_component
        ]
    )

demo.launch(share=True, server_port=7861, server_name='0.0.0.0')