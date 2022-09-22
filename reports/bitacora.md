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

- <https://resources.unity.com/ai-ml-content/sample-home-datasets>


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













