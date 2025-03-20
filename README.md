<h1 text="bold">ESPAÑOL:</h1>

<h2 text="bold">Perceptrón Multicapa para Clasificación de Colores</h2>
Este proyecto implementa un perceptrón multicapa (PML) en C# para la clasificación de colores basados en sus valores RGB. El sistema permite entrenar la red neuronal con un conjunto de datos predefinidos y luego utilizar el modelo entrenado para predecir el color de un píxel seleccionado de una imagen.

<h2 text="bold">Características Principales</h2>

<ul> 
<li>Clasificación de Colores: El PML es capaz de clasificar colores en 12 categorías diferentes: Rojo, Verde, Azul, Amarillo, Rosa, Naranja, Morado, Cyan, Gris, Café, Negro y Blanco.</li><br>

<li>Entrenamiento Personalizado: El sistema permite entrenar la red neuronal con un conjunto de datos predefinidos o cargar un modelo previamente entrenado.</li><br>

<li>Interfaz Gráfica: La aplicación cuenta con una interfaz gráfica que permite cargar imágenes, seleccionar píxeles y visualizar la predicción del color.</li><br>

<li>Persistencia de Datos: Los pesos y sesgos de la red neuronal pueden ser guardados y cargados desde archivos de texto, lo que permite reutilizar el modelo entrenado en sesiones futuras.</li><br>
</ul>

<h2 text="bold">Estructura del Proyecto</h2>
El proyecto está organizado en varias clases principales:

<ul>
<li>VariablesGlobales: Contiene las variables globales utilizadas en todo el sistema, como rutas de archivos, parámetros de entrenamiento y datos de entrada/salida.</li><br>

<li>Archivos: Maneja la lectura y escritura de archivos, incluyendo la creación, limpieza y escritura de archivos de configuración y datos.</li><br>

<li>Form1: Es la clase principal de la interfaz gráfica, que maneja la interacción del usuario, la carga de imágenes, la selección de píxeles y la visualización de resultados.</li><br>

<li>PML: Implementa el perceptrón multicapa, incluyendo métodos para el entrenamiento, propagación, retropropagación y normalización de datos.</li><br>
</ul>

<h2 text="bold">Requisitos</h2>
<ul>
<li>.NET Framework: El proyecto está desarrollado en C# y requiere .NET Framework para su ejecución.</li><br>

<li>Visual Studio: Se recomienda utilizar Visual Studio para abrir y compilar el proyecto.</li><br>
</ul>
<h2 text="bold">Instrucciones de Uso</h2>
<ol>
<li>Compilación: Abre el proyecto en Visual Studio y compílalo.</li><br>

<li>Ejecución: Ejecuta la aplicación y se abrirá la interfaz gráfica.</li><br>

<li>Carga de Imágenes: Utiliza los botones para cargar una imagen predefinida o selecciona una imagen personalizada desde tu computadora.</li><br>

<li>Selección de Píxeles: Haz clic en cualquier parte de la imagen para seleccionar un píxel. El sistema mostrará los valores RGB del píxel seleccionado.</li><br>

<li>Entrenamiento: Si no has cargado un modelo previamente entrenado, puedes entrenar la red neuronal utilizando el botón correspondiente.</li><br>

<li>Predicción: Una vez entrenado el modelo, haz clic en el botón "Probar" para predecir el color del píxel seleccionado.</li><br>

<li>Guardar/Cargar Modelo: Puedes guardar el modelo entrenado para usarlo en sesiones futuras o cargar un modelo previamente guardado.</li><br>
</ol>
<h2 text="bold">Archivos de Configuración y Datos</h2>
<ul>
<li>configuracion.txt: Contiene los pesos y sesgos de la red neuronal.</li><br>

<li>colores.txt: Almacena los datos de los píxeles seleccionados por el usuario.</li><br>
</ul>
<h2 text="bold">Consideraciones</h2>
<ul>
<li>Precisión: La precisión del modelo depende del conjunto de datos de entrenamiento y de los parámetros configurados (tasa de aprendizaje, número de épocas, etc.).</li><br>

<li>Rendimiento: El entrenamiento de la red neuronal puede ser computacionalmente costoso, especialmente con un gran número de épocas.</li><br>
</ul>
<h2 text="bold">Contribuciones</h2>
<h3 text="bold">Si deseas contribuir a este proyecto, siéntete libre de hacer un fork y enviar un pull request con tus mejoras.</h3>


<h1 text="bold">ENGLISH:</h1>

<h2 text="bold">Multilayer Perceptron for Color Classification</h2>
This project implements a multilayer perceptron (MLP) in C# for color classification based on their RGB values. The system allows to train the neural network with a predefined data set and then use the trained model to predict the color of a selected pixel of an image.

<h2 text="bold">Main Features</h2>
<ul>
<li>Color Classification: The MLP is capable of classifying colors into 12 different categories: Red, Green, Blue, Yellow, Pink, Orange, Purple, Cyan, Gray, Brown, Black and White.</li>

<li>Custom Training: The system allows to train the neural network with a predefined data set or to load a previously trained model.</li>

<li>Graphical Interface: The application has a graphical interface that allows to load images, select pixels and visualize the color prediction.</li>

<li>Data Persistence: The weights and biases of the neural network can be saved and loaded from text files, allowing to reuse the trained model in future sessions.</li>
</ul>
<h2 text="bold">Project Structure</h2>
The project is organized into several main classes:
<ul>
<li>GlobalVariables: Contains the global variables used throughout the system, such as file paths, training parameters, and input/output data.</li>

<li>Files: Handles reading and writing files, including creating, cleaning, and writing configuration and data files.</li>

<li>Form1: This is the main class for the graphical interface, which handles user interaction, image loading, pixel selection, and displaying results.</li>

<li>PML: Implements the multilayer perceptron, including methods for training, propagation, backpropagation, and data normalization.</li>
</ul>
<h2 text="bold">Requirements</h2>
<ul>
<li>.NET Framework: The project is developed in C# and requires .NET Framework to run.</li>

<li>Visual Studio: It is recommended to use Visual Studio to open and compile the project.</li>
</ul>
<h2 text="bold">Instructions for Use</h2>
<ul>
<li>Compilation: Open the project in Visual Studio and compile it.</li>

<li>Execution: Run the application and the graphical interface will open.</li>

<li>Loading Images: Use the buttons to load a predefined image or select a custom image from your computer.</li>

<li>Pixel Selection: Click anywhere on the image to select a pixel. The system will display the RGB values ​​of the selected pixel.</li>

<li>Training: If you have not loaded a pre-trained model, you can train the neural network using the corresponding button.</li>

<li>Prediction: Once the model is trained, click the "Test" button to predict the color of the selected pixel.</li>

<li>Save/Load Model: You can save the trained model for use in future sessions or load a previously saved model.</li>
</ul>
<h2 text="bold">Configuration and Data Files</h2>
<ul>
<li>configuration.txt: Contains the weights and biases of the neural network.</li>

<li>colors.txt: Stores the data for the pixels selected by the user.</li>
</ul>
<h2 text="bold">Considerations</h2>
<ul>
<li>Accuracy: The accuracy of the model depends on the training dataset and the configured parameters (learning rate, number of epochs, etc.).</li>

<li>Performance: Training the neural network can be computationally expensive, especially with a large number of epochs.</li>
</ul>
<h2 text="bold">Contributions</h2>
If you would like to contribute to this project, feel free to fork it and submit a pull request with your improvements.
