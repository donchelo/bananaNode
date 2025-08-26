import os
import io
import base64
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

def load_config_file():
    """Busca y carga config.env desde múltiples ubicaciones"""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), 'config.env'),
        os.path.join(os.path.dirname(__file__), '.env'),
        os.path.expanduser('~/gemini_config.env'),
        '/workspace/ComfyUI/custom_nodes/gemini_flash_image/config.env',
    ]
    
    print("🔍 Buscando configuración de API key para Gemini Flash Image...")
    
    for config_path in possible_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                            print(f"✅ Variable {key.strip()} configurada desde: {config_path}")
                print(f"✅ Config cargado desde: {config_path}")
                return True
            except Exception as e:
                print(f"⚠️ Error leyendo {config_path}: {e}")
                continue
    
    if os.getenv('GOOGLE_API_KEY'):
        print("✅ Usando variables de entorno del sistema")
        return True
        
    print("⚠️ No se encontró configuración de API key")
    print("🔍 Rutas buscadas:")
    for path in possible_paths:
        print(f"   - {path} {'✅' if os.path.exists(path) else '❌'}")
    return False

class GeminiFlashImage:
    """
    Nodo personalizado para ComfyUI que integra Gemini 2.5 Flash Image
    Capacidades: generación, edición, consistencia de personajes, fusión multi-imagen
    """
    
    def __init__(self):
        self.client = None
        self.api_key = None
        # Cargar configuración
        load_config_file()
        self.initialize_client()
    
    def initialize_client(self):
        """Inicializa el cliente de Google Generative AI"""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ Error: GOOGLE_API_KEY no encontrada en variables de entorno")
            print("💡 Soluciones:")
            print("   1. Crear archivo config.env en el directorio del custom node")
            print("   2. Configurar variable de entorno GOOGLE_API_KEY")
            print("   3. Obtener API key en: https://aistudio.google.com/app/apikey")
            return
        
        try:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel('gemini-2.5-flash-image-preview')
            print("✅ Cliente Gemini Flash Image inicializado correctamente")
        except Exception as e:
            print(f"❌ Error inicializando cliente Gemini: {e}")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Generate a beautiful landscape with mountains and a lake"
                }),
                "mode": ([
                    "generate",
                    "edit", 
                    "character_consistency",
                    "multi_image_fusion",
                    "prompt_based_editing",
                    "world_knowledge",
                    "custom"
                ], {"default": "generate"}),
                "model": (["gemini-2.5-flash-image-preview"], {"default": "gemini-2.5-flash-image-preview"}),
                "use_env_key": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "primary_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "secondary_image": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "text_response", "debug_info")
    FUNCTION = "generate_with_gemini"
    CATEGORY = "Gemini/Image"
    DESCRIPTION = "Generate and edit images using Gemini 2.5 Flash Image (nano-banana)"
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convierte tensor de ComfyUI a PIL Image"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # Remover dimensión batch
        
        # Convertir de tensor [H, W, C] a numpy
        np_image = tensor.cpu().numpy()
        
        # Asegurar valores en rango [0, 255]
        if np_image.max() <= 1.0:
            np_image = np_image * 255.0
        
        np_image = np_image.astype(np.uint8)
        
        # Convertir a PIL Image
        if np_image.shape[2] == 3:  # RGB
            return Image.fromarray(np_image, 'RGB')
        elif np_image.shape[2] == 4:  # RGBA
            return Image.fromarray(np_image, 'RGBA')
        else:
            # Manejar escala de grises
            return Image.fromarray(np_image[:,:,0], 'L')
    
    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Convierte PIL Image a tensor de ComfyUI"""
        # Asegurar formato RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convertir a numpy array
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        
        # Convertir a tensor con dimensión batch [1, H, W, C]
        tensor = torch.from_numpy(np_image)[None,]
        
        return tensor
    
    def get_optimized_prompt(self, mode: str, custom_prompt: str) -> str:
        """Obtiene prompts optimizados según el modo seleccionado"""
        mode_prompts = {
            "generate": custom_prompt,
            "edit": f"Edit the provided image based on this request: {custom_prompt}. Maintain the overall composition and only change what's specifically requested.",
            "character_consistency": f"Maintain the exact same character/subject from the reference image and {custom_prompt}. Preserve all facial features, clothing, distinctive characteristics, and personality while adapting to the new context.",
            "multi_image_fusion": f"Seamlessly blend and merge elements from the provided images to create: {custom_prompt}. Maintain realistic lighting, perspective, and natural composition.",
            "prompt_based_editing": f"Make these specific targeted changes to the image: {custom_prompt}. Use precise local edits while preserving all other elements exactly as they are. Maintain photorealistic quality.",
            "world_knowledge": f"Using your comprehensive knowledge of the real world, {custom_prompt}. Ensure factual accuracy, realistic representation, and attention to real-world details.",
            "custom": custom_prompt
        }
        
        return mode_prompts.get(mode, custom_prompt)
    
    def prepare_contents(self, prompt: str, primary_image=None, reference_image=None, 
                        secondary_image=None, mask_image=None) -> List[Any]:
        """Prepara el contenido para la API de Gemini"""
        contents = [prompt]
        
        # Agregar imágenes en orden de prioridad
        images_to_process = [
            primary_image, reference_image, secondary_image, mask_image
        ]
        
        for image_tensor in images_to_process:
            if image_tensor is not None:
                try:
                    pil_image = self.tensor_to_pil(image_tensor)
                    contents.append(pil_image)
                except Exception as e:
                    print(f"⚠️ Error procesando imagen: {e}")
                    continue
        
        return contents
    
    def process_response(self, response) -> Tuple[Optional[Image.Image], str]:
        """Procesa la respuesta de Gemini y extrae imagen y texto"""
        generated_image = None
        text_response = ""
        
        try:
            if hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                
                if hasattr(candidate, 'content') and candidate.content.parts:
                    for part in candidate.content.parts:
                        # Extraer texto
                        if hasattr(part, 'text') and part.text:
                            text_response += part.text + " "
                        
                        # Extraer imagen
                        elif hasattr(part, 'inline_data') and part.inline_data:
                            image_data = part.inline_data.data
                            generated_image = Image.open(io.BytesIO(image_data))
            
            # Limpiar texto
            text_response = text_response.strip()
            if not text_response:
                text_response = "Imagen generada exitosamente con Gemini 2.5 Flash Image"
                
        except Exception as e:
            text_response = f"Error procesando respuesta: {str(e)}"
            print(f"❌ Error en process_response: {e}")
        
        return generated_image, text_response
    
    def generate_with_gemini(self, prompt: str, mode: str, model: str, use_env_key: bool = True,
                           primary_image=None, reference_image=None, secondary_image=None, 
                           mask_image=None, api_key: str = "") -> Tuple:
        """Función principal para generar/editar imágenes con Gemini"""
        debug_info = []
        
        # Validar configuración de API
        if use_env_key:
            if not self.client:
                error_msg = "Cliente Gemini no inicializado. Configura GOOGLE_API_KEY en config.env"
                return (torch.zeros(1, 512, 512, 3), f"Error: {error_msg}", error_msg)
        else:
            if not api_key.strip():
                error_msg = "API key requerida cuando use_env_key está desactivado"
                return (torch.zeros(1, 512, 512, 3), f"Error: {error_msg}", error_msg)
            
            try:
                genai.configure(api_key=api_key.strip())
                temp_client = genai.GenerativeModel(model)
            except Exception as e:
                error_msg = f"Error configurando API key temporal: {str(e)}"
                return (torch.zeros(1, 512, 512, 3), f"Error: {error_msg}", error_msg)
            
            client_to_use = temp_client
        
        if use_env_key:
            client_to_use = self.client
        
        try:
            # Obtener prompt optimizado
            final_prompt = self.get_optimized_prompt(mode, prompt)
            debug_info.append(f"Modo: {mode}")
            debug_info.append(f"Modelo: {model}")
            debug_info.append(f"Prompt optimizado: {final_prompt[:100]}...")
            
            # Preparar contenido
            contents = self.prepare_contents(
                final_prompt, primary_image, reference_image, 
                secondary_image, mask_image
            )
            
            image_count = len([c for c in contents if isinstance(c, Image.Image)])
            debug_info.append(f"Imágenes de entrada: {image_count}")
            
            # Configurar parámetros de seguridad más permisivos
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            # Realizar llamada a la API
            print("🚀 Enviando solicitud a Gemini 2.5 Flash Image...")
            response = client_to_use.generate_content(
                contents=contents,
                safety_settings=safety_settings,
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 40,
                    'max_output_tokens': 2048,
                }
            )
            
            # Procesar respuesta
            generated_image, text_response = self.process_response(response)
            
            if generated_image:
                result_tensor = self.pil_to_tensor(generated_image)
                debug_info.append(f"Imagen generada: {generated_image.size}")
                debug_info.append("✅ Éxito: Generación completada")
            else:
                # Si no hay imagen, crear una imagen placeholder
                placeholder = Image.new('RGB', (512, 512), color=(100, 100, 100))
                result_tensor = self.pil_to_tensor(placeholder)
                debug_info.append("⚠️ No se generó imagen, usando placeholder")
                text_response = text_response or "No se pudo generar imagen"
            
            # Agregar información de costo
            debug_info.append("💰 Costo estimado: ~$0.039 por imagen")
            debug_info.append("🔒 Imagen incluye marca SynthID invisible")
            
            debug_str = " | ".join(debug_info)
            
            return (result_tensor, text_response, debug_str)
            
        except Exception as e:
            error_msg = f"Error en generación Gemini: {str(e)}"
            debug_info.append(error_msg)
            debug_str = " | ".join(debug_info)
            print(f"❌ {error_msg}")
            
            # Retornar imagen en negro en caso de error
            error_image = torch.zeros(1, 512, 512, 3)
            return (error_image, f"Error: {str(e)}", debug_str)

# Registro del nodo para ComfyUI
NODE_CLASS_MAPPINGS = {
    "CheLogarchoGemini": GeminiFlashImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheLogarchoGemini": "CheLogarcho Gemini Flash"
}