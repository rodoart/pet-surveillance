# 23-junio-2022

## Iniciando la bítacora

***Autor: Rodolfo***

- Hoy recibimos un correo electrónico con indicaciones para escribir una bítacora.

- El proyecto ahora está en GitHub, desde ahí trabajaremos en equipo.

- Actualmente reviso los temas de la Fase 2: _Visual Thinking_.
- También estoy configurando un notebook de prueba `semantic_segmentation` para probar si este tipo de algoritmos funciona para nuestro propósito.


# 24-junio-2022

## Segmentación semántica revisada.

***Autor: Rodolfo***

Existen varias opciones pre-entrenadas de redes neuronales de keras para la segmentación semántica. Estoy siguiendo esta:

- [Image segmentation](https://github.com/divamgupta/image-segmentation-keras)

Hoy voy a tratar de entrenar a una red neuornal con imágenes de habitaciones creadas artificialmente, cortesía de :

- <https://resources.unity.com/ai-ml-content/sample-home-datasets>(https://github.com/sithu31296/semantic-segmentation)




# 2-julio-2022
***Autor: Rodolfo***

Se realizó un diagrama con el procesamiento del video, esta pendiente completar las fases de desarrollo, entrenamiento, y despliegue.

Se están evaluando distintas redes neuronales preentrenadas para la segmentación semántica.

# 13-julio-2022
***Autor: Rodolfo***

Se creo un prototipo animado mediante un gif del resultado final.
Se creo un modulo make_dataset para la descarga y ordenamiento de los datos de entrenamiento y validación.


# 14-julio-2022
***Autor: Rodolfo***

En el archivo `0.1-vigilancia_mascotas-semantic_semantic_segmentation_test_with_room_data.ipynb` se probaron con las imágenes del dataset [de interiores](https://resources.unity.com/ai-ml-content/sample-home-datasets) tres redes neuronales preentrenadas de segmentación automática:

- pspnet_50_ADE_20K
- pspnet_101_cityscapes
- pspnet_101_voc12

Se determinó que la mejor para este proyecto es pspnet_101_voc12, pues aún sin hacer ajustes tiene una precisión mean_IU bastante elevada (0.043). 

También se hizo un modulo `make_dataset` que automáticamente descarga y ordena este dataset, para futuras pruebas.

# 15-julio-2022
***Autor: Rodolfo***

Se descubrió un problema con las imágenes "labels" del dataset unity, las imágenes no estaban correctamente formateadas. Se deben formatear de forma correcta y reiniciar las pruebas.





# 25-julio-2022 
***Autor: Rodolfo***

Luego de múltiples pruebas con la librería [Image segmentation de divamgupta](https://github.com/divamgupta/image-segmentation-keras). La precisión lograda fue ínfima, y al evaluar con imágenes muestra, no está segmentando de forma correcta. Se propone buscar nuevas librerías.

# 1-agosto-2022
***Autor: Rodolfo***

Entre las librerías investigadas se probó mediante pruebas cualitativas con las imágenes del dataset que la mejor librería segmentando para nuestros propósitos es [SOTA Semantic Segmentation Models](https://github.com/sithu31296/semantic-segmentation) de Sithu Aung. Las imágenes muestra segmentadas se pueden ver `data\processed\semantic_segmentation\unity_residential_interiors\train_images\7.png`.

La única desventaja se que es una libería de Torch en lugar de Keras, que tiene una más amplia implementación y distribución.


# 3-agosto-2022
***Autor: Rodolfo***

Dentro de las opciones pre-entrenadas de [SOTA Semantic Segmentation Models](https://github.com/sithu31296/semantic-segmentation), la más recomendada es Segformer, que se emplea para segmentación semántica general. Funciona bien con las imágenes de interiores


# 15-agosto-2022
***Autor: Rodolfo***

Se creó el modulo `pet_surveillance\models\segformer.py`que contiene un objeto que permite descargar e instalar las dependencias necesarias  [SOTA Semantic Segmentation Models](https://github.com/sithu31296/semantic-segmentation) de forma automática. Además genera imágenes segmentadas.



# 17-agosto-2022
***Autor: Rodolfo***

Se hizo una versión preliminar del detector de piso, en el que simplemente se elige el objeto segmentado de mayor tamaño y que además toca el borde inferior de la imagen.


# 6-septiembre-2022
***Autor: Rodolfo***

Se intenta mejorar el algoritmo de detección del piso. El problema que hay que resolver consiste en eligir dentro de todos los objetos segmentados el objeto que corresponde el piso de forma correcta y consistente.

# 8-septiembre-2022
***Autor: Rodolfo***

Las solución encontrada empírica fue calcular la probabilidad de que un objeto se encuentre en cerca del piso de la siguiente manera:

$$
P_{\text{piso}} = p_\text{{inferior}}\ p_\text{{tamaño}} \ p_\text{{distribución \ vertical}}
$$


Donde:

$p_\text{{inferior}} = \frac{\text{píxeles del objeto en el borde inferior}}{w}$ 

Es la fracción de píxeles del objeto en el borde inferior respecto a $w$ ancho total de la imagen en píxeles.

$p_\text{{tamaño}} = \frac{A_{\text{objeto}}}{A_{\text{total}}}$ 

Es la fracción de la suma total de píxeles del objeto respecto al área total de la imagen en píxeles.


$p_\text{{distribución}} = \frac{y_{\text{CM}}}{h}$

$y_{\text{CM}}$ es la componente vertical del calculo del centroide del objeto sobre la altura $h$ total de la imagen en píxeles. Entre más píxeles del objeto estén en la parte inferior, más bajo será su centroide.


$$

y_{\text{CM}} = \sum_{j=0}^{h-1}\frac{m_j \cdot (j+1)}{A_{\text{objeto}}}

$$

donde

$j$ es la coordenada vertical de la matriz de la imagen, cada iteración de la suma está corriendo una línea de la imagen.

$m_j$ es el total de píxeles en la línea $j$.


El objeto con mayor $P_{\text{piso}}$ tiene mayor probabilidad de ser el piso. Las $p$s se calculan en el orden de la fórmula, realmente la única que puede ser 0 es la $p_\text{{inferior}}$ por lo que si esto pasa, el objeto queda descartado y ya no se calculan las otras. 

Esto se programó como el método `Segformer.detect_floor` del módulo `pet_surveillance\models\segformer.py`, que devuelve una matriz booleana del tamaño de la imagen proporcionada, en la que los valores verdaderos representan el piso detectado.





# 9-septiembre-2022
***Autor: Rodolfo***

Se implementó la detección del piso dentro del programa de captura de vídeo. Se creó la función `boolean_mask_overlay`pet_surveillance\utils\video_layers.py` 

Dada una imagen y su matriz de piso, colore de forma transparente el piso detectado, que se utiliza en el vídeo.



