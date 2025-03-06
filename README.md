# PML

This is a basic PML to identified selected colors, this program was created to be used in a project that I'm working on.
And I'm sharing it with you, so you can use it in your projects too.
So I'm want to get more experience at this area, so I'm open to suggestions and feedbacks.


ESPAÑOL:

**Perceptrón Multicapa para Clasificación de Colores**
Este proyecto implementa un perceptrón multicapa (PML) en C# para la clasificación de colores basados en sus valores RGB. El sistema permite entrenar la red neuronal con un conjunto de datos predefinidos y luego utilizar el modelo entrenado para predecir el color de un píxel seleccionado de una imagen.

**Características Principales**
Clasificación de Colores: El PML es capaz de clasificar colores en 12 categorías diferentes: Rojo, Verde, Azul, Amarillo, Rosa, Naranja, Morado, Cyan, Gris, Café, Negro y Blanco.

Entrenamiento Personalizado: El sistema permite entrenar la red neuronal con un conjunto de datos predefinidos o cargar un modelo previamente entrenado.

Interfaz Gráfica: La aplicación cuenta con una interfaz gráfica que permite cargar imágenes, seleccionar píxeles y visualizar la predicción del color.

Persistencia de Datos: Los pesos y sesgos de la red neuronal pueden ser guardados y cargados desde archivos de texto, lo que permite reutilizar el modelo entrenado en sesiones futuras.

**Estructura del Proyecto**
El proyecto está organizado en varias clases principales:

VariablesGlobales: Contiene las variables globales utilizadas en todo el sistema, como rutas de archivos, parámetros de entrenamiento y datos de entrada/salida.

Archivos: Maneja la lectura y escritura de archivos, incluyendo la creación, limpieza y escritura de archivos de configuración y datos.

Form1: Es la clase principal de la interfaz gráfica, que maneja la interacción del usuario, la carga de imágenes, la selección de píxeles y la visualización de resultados.

PML: Implementa el perceptrón multicapa, incluyendo métodos para el entrenamiento, propagación, retropropagación y normalización de datos.

**Requisitos**
.NET Framework: El proyecto está desarrollado en C# y requiere .NET Framework para su ejecución.

Visual Studio: Se recomienda utilizar Visual Studio para abrir y compilar el proyecto.

**Instrucciones de Uso**
Compilación: Abre el proyecto en Visual Studio y compílalo.

Ejecución: Ejecuta la aplicación y se abrirá la interfaz gráfica.

Carga de Imágenes: Utiliza los botones para cargar una imagen predefinida o selecciona una imagen personalizada desde tu computadora.

Selección de Píxeles: Haz clic en cualquier parte de la imagen para seleccionar un píxel. El sistema mostrará los valores RGB del píxel seleccionado.

Entrenamiento: Si no has cargado un modelo previamente entrenado, puedes entrenar la red neuronal utilizando el botón correspondiente.

Predicción: Una vez entrenado el modelo, haz clic en el botón "Probar" para predecir el color del píxel seleccionado.

Guardar/Cargar Modelo: Puedes guardar el modelo entrenado para usarlo en sesiones futuras o cargar un modelo previamente guardado.

**Archivos de Configuración y Datos**
configuracion.txt: Contiene los pesos y sesgos de la red neuronal.

colores.txt: Almacena los datos de los píxeles seleccionados por el usuario.

**Consideraciones**
Precisión: La precisión del modelo depende del conjunto de datos de entrenamiento y de los parámetros configurados (tasa de aprendizaje, número de épocas, etc.).

Rendimiento: El entrenamiento de la red neuronal puede ser computacionalmente costoso, especialmente con un gran número de épocas.

**Contribuciones**
Si deseas contribuir a este proyecto, siéntete libre de hacer un fork y enviar un pull request con tus mejoras.


ENGLISH:

**Multilayer Perceptron for Color Classification**
This project implements a multilayer perceptron (MLP) in C# for color classification based on their RGB values. The system allows to train the neural network with a predefined data set and then use the trained model to predict the color of a selected pixel of an image.

**Main Features**
Color Classification: The MLP is capable of classifying colors into 12 different categories: Red, Green, Blue, Yellow, Pink, Orange, Purple, Cyan, Gray, Brown, Black and White.

Custom Training: The system allows to train the neural network with a predefined data set or to load a previously trained model.

Graphical Interface: The application has a graphical interface that allows to load images, select pixels and visualize the color prediction.

Data Persistence: The weights and biases of the neural network can be saved and loaded from text files, allowing to reuse the trained model in future sessions.

**Project Structure**
The project is organized into several main classes:

GlobalVariables: Contains the global variables used throughout the system, such as file paths, training parameters, and input/output data.

Files: Handles reading and writing files, including creating, cleaning, and writing configuration and data files.

Form1: This is the main class for the graphical interface, which handles user interaction, image loading, pixel selection, and displaying results.

PML: Implements the multilayer perceptron, including methods for training, propagation, backpropagation, and data normalization.

**Requirements**
.NET Framework: The project is developed in C# and requires .NET Framework to run.

Visual Studio: It is recommended to use Visual Studio to open and compile the project.

**Instructions for Use**
Compilation: Open the project in Visual Studio and compile it.

Execution: Run the application and the graphical interface will open.

Loading Images: Use the buttons to load a predefined image or select a custom image from your computer.

Pixel Selection: Click anywhere on the image to select a pixel. The system will display the RGB values ​​of the selected pixel.

Training: If you have not loaded a pre-trained model, you can train the neural network using the corresponding button.

Prediction: Once the model is trained, click the "Test" button to predict the color of the selected pixel.

Save/Load Model: You can save the trained model for use in future sessions or load a previously saved model.

**Configuration and Data Files**
configuration.txt: Contains the weights and biases of the neural network.

colors.txt: Stores the data for the pixels selected by the user.

**Considerations**
Accuracy: The accuracy of the model depends on the training dataset and the configured parameters (learning rate, number of epochs, etc.).

Performance: Training the neural network can be computationally expensive, especially with a large number of epochs.

**Contributions**
If you would like to contribute to this project, feel free to fork it and submit a pull request with your improvements.
