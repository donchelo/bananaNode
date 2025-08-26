# Gemini 2.5 Flash Image - Custom Node para ComfyUI

## 📋 Descripción
Este custom node integra **Gemini 2.5 Flash Image** (también conocido como "nano-banana") directamente en ComfyUI, proporcionando capacidades avanzadas de generación y edición de imágenes con IA.

## 🌟 Características Principales
- **🎨 Generación de imágenes de alta calidad**: Estado del arte en calidad visual
- **✏️ Edición basada en prompts**: Modificaciones precisas usando lenguaje natural
- **👤 Consistencia de personajes**: Mantiene personajes idénticos entre imágenes
- **🔄 Fusión multi-imagen**: Combina múltiples imágenes en una composición
- **🌍 Conocimiento del mundo**: Comprensión semántica profunda del mundo real
- **🔒 SynthID**: Marca de agua invisible automática en todas las imágenes
- **💰 Costo eficiente**: $0.039 por imagen generada

## 🚀 Instalación

### 1. Copiar Archivos
Copia la carpeta completa a tu directorio de custom nodes:
```bash
# Ubicación típica
ComfyUI/custom_nodes/gemini_flash_image/
```

### 2. Instalar Dependencias
```bash
cd ComfyUI/custom_nodes/gemini_flash_image
pip install -r requirements.txt
```

**Para ComfyUI Portable:**
```bash
cd ComfyUI_windows_portable
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\gemini_flash_image\requirements.txt
```

### 3. Configurar API Key
1. Obtén tu API key en: https://aistudio.google.com/app/apikey
2. Edita el archivo `config.env`:
```bash
GOOGLE_API_KEY=tu_clave_api_real_aqui
```

### 4. Reiniciar ComfyUI
Reinicia ComfyUI para que detecte el nuevo nodo.

## 🎯 Modos de Operación

### 🎨 **generate** - Generación de Imágenes
Genera imágenes completamente nuevas basadas en prompts de texto.

**Entradas:**
- `prompt`: Descripción de la imagen a generar
- `primary_image`: (Opcional) Imagen de referencia

**Ejemplo:**
```
"Un gato siamés sentado en un jardín japonés al atardecer, estilo fotográfico"
```

### ✏️ **edit** - Edición de Imágenes
Modifica una imagen existente manteniendo la composición general.

**Entradas:**
- `prompt`: Instrucciones de edición
- `primary_image`: Imagen a editar

**Ejemplo:**
```
"Cambia el color del vestido a azul y agrega un sombrero"
```

### 👤 **character_consistency** - Consistencia de Personajes
Mantiene personajes idénticos entre diferentes imágenes.

**Entradas:**
- `prompt`: Nueva escena para el personaje
- `reference_image`: Imagen con el personaje de referencia

**Ejemplo:**
```
"El mismo personaje en una playa tropical"
```

### 🔄 **multi_image_fusion** - Fusión Multi-Imagen
Combina elementos de múltiples imágenes en una composición.

**Entradas:**
- `prompt`: Descripción de la composición final
- `primary_image`: Imagen principal
- `secondary_image`: Imagen secundaria para fusionar

**Ejemplo:**
```
"Combina el paisaje de la primera imagen con el personaje de la segunda"
```

### 🎯 **prompt_based_editing** - Edición Basada en Prompts
Realiza ediciones específicas y localizadas.

**Entradas:**
- `prompt`: Cambios específicos a realizar
- `primary_image`: Imagen a editar
- `mask_image`: (Opcional) Máscara de área a editar

**Ejemplo:**
```
"Reemplaza el fondo con un bosque mágico"
```

### 🌍 **world_knowledge** - Conocimiento del Mundo
Utiliza el conocimiento semántico profundo del modelo.

**Entradas:**
- `prompt`: Solicitud que requiere conocimiento del mundo real

**Ejemplo:**
```
"Una escena histórica de la antigua Roma con arquitectura auténtica"
```

### ⚙️ **custom** - Modo Personalizado
Permite prompts completamente personalizados.

**Entradas:**
- `prompt`: Cualquier prompt personalizado

## 🔧 Configuración Avanzada

### Parámetros del Nodo

| Parámetro | Tipo | Descripción | Valor por Defecto |
|-----------|------|-------------|-------------------|
| `prompt` | STRING | Descripción de la imagen o instrucciones | "Generate a beautiful landscape..." |
| `mode` | LIST | Modo de operación | "generate" |
| `model` | LIST | Modelo de Gemini a usar | "gemini-2.5-flash-image-preview" |
| `use_env_key` | BOOLEAN | Usar API key del archivo config.env | True |
| `primary_image` | IMAGE | Imagen principal (opcional) | - |
| `reference_image` | IMAGE | Imagen de referencia (opcional) | - |
| `secondary_image` | IMAGE | Imagen secundaria (opcional) | - |
| `mask_image` | IMAGE | Máscara de edición (opcional) | - |
| `api_key` | STRING | API key manual (opcional) | "" |

### Salidas del Nodo

| Salida | Tipo | Descripción |
|--------|------|-------------|
| `image` | IMAGE | Imagen generada o editada |
| `text_response` | STRING | Respuesta de texto de Gemini |
| `debug_info` | STRING | Información de depuración |

## 💡 Ejemplos de Uso

### Ejemplo 1: Generación Simple
```
Prompt: "Un dragón dorado volando sobre montañas nevadas al amanecer"
Modo: generate
```

### Ejemplo 2: Edición de Personaje
```
Prompt: "Cambia el color del cabello a rojo y agrega gafas de sol"
Modo: edit
Primary Image: [tu imagen]
```

### Ejemplo 3: Consistencia de Personaje
```
Prompt: "El mismo personaje en una oficina moderna"
Modo: character_consistency
Reference Image: [imagen con el personaje]
```

## 🛠️ Solución de Problemas

### Error: "GOOGLE_API_KEY no encontrada"
1. Verifica que el archivo `config.env` existe en el directorio del nodo
2. Asegúrate de que la API key esté correctamente configurada
3. Reinicia ComfyUI después de cambiar la configuración

### Error: "Cliente Gemini no inicializado"
1. Verifica tu conexión a internet
2. Confirma que tu API key es válida
3. Revisa los logs de ComfyUI para más detalles

### Imagen no se genera
1. Verifica que el prompt sea claro y específico
2. Asegúrate de que las imágenes de entrada sean válidas
3. Revisa la información de debug en la salida del nodo

## 📊 Información de Costos

- **Costo por imagen**: ~$0.039 USD
- **Límites**: Según tu plan de Google AI Studio
- **Marca de agua**: SynthID incluida automáticamente

## 🔗 Enlaces Útiles

- [Google AI Studio](https://aistudio.google.com/app/apikey) - Obtener API key
- [Documentación de Gemini](https://ai.google.dev/docs) - Documentación oficial
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Framework principal

## 📝 Licencia

Este proyecto está bajo la licencia MIT. Ver archivo LICENSE para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request en el repositorio.

---

**Nota**: Este nodo requiere una API key válida de Google AI Studio para funcionar. Asegúrate de configurar tu clave antes de usar el nodo.