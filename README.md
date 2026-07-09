# Prueba de los modelos xgboost y randomforest
Se ha descartado el modelo regresion logisitca por ser ineficiente a comparacion de xgboost y random forest

## Columnas tomados para el entrenamiento y prediccion

* Oferta Educativa
* Grado
* Genero
* Promedio Final
* Promedio Cuatrimestre Anterior
* Promedio Bachillerato

## Columnas codificadas para pasar de texto a binario(label_encoders.pkl)

* Oferta Educativa
* Grado
* Genero

## Remplazo de csv para entrenamiento

Si se desea cambiar el csv de entrenamiento, se debe ingresar el archivo dentro de la carpeta del modelo a elegir. Si tiene un nombre distinto, se debe reemplazar el nombre en el código de entrenamiento.

### Comando de entrenamiento del modelo
* random forest

python randomforest/entrenamiento.py

* xgboost

python xgboost/entrenamiento.py

## Prediccion

Para usar el modelo entrenado es necesario usar los dos achivos dentro de la carpeta de los modelos, una contiene el modelo ya entrenado y la otra los caracteres codificado de las columnas del csv.