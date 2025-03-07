using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Perceptron_Multicapa_Colores
{
	/// <summary>
	/// Clase para manejar las variables globales las cuales se usarán a lo largo del sistema.
	/// </summary>
	public class VariablesGlobales
	{
		/// <summary>
		/// Escritorio: Ruta de cada entorno al escritorio.
		/// Carpeta: Nombre de la carpeta de archivos para guardar los archivos.
		/// Ruta: Combina la ruta del escritorio con la carpeta donde se guardarán los archivos.
		/// FormatoArchivos: Es el formato en el que se almacenaran los archivos.
		/// Configuracion: Nombre del archivo para guardar la configuración.
		/// Datos: Nombre del archivo en el que se almacenarán los datos.
		/// </summary>
		private static readonly string Escritorio = Environment.GetFolderPath(Environment.SpecialFolder.Desktop), 
			Carpeta = @"\Archivos\", Ruta = Path.Combine(Escritorio + Carpeta), FormatoArchivos = ".txt",
			Configuracion = "configuracion", Datos = "colores";

		/// <summary>
		/// Epocas: Iteraciones que se harán para el aprendizaje.
		/// Min: Valor mínimo dentro del contexto de los datos.
		/// Max: Valor máximo dentro del contexto de los datos.
		/// </summary>
		private static readonly int Epocas = 100000, Min = 0, Max = 255;

		/// <summary>
		/// TasaAprendizaje: Factor de aprendizaje.
		/// ErrorMinimo: Error mínimo el cual se espera alcanzar.
		/// </summary>
		private static readonly double TasaAprendizaje = 0.001f, ErrorMinimo = 0.0001f;

		/// <summary>
		/// DatosPerceptron: Datos que se utilizarán para el perceptrón multicapa.
		/// {Número de datos que va
		/// n a entrar, Número de neuronas en cada capa oculta (pueden ser n número de capas ocultas), Número de datos que van a salir }
		/// </summary>
		private static readonly int[] DatosPerceptron = { 3, 10, 12 };

		/// <summary>
		/// NombresColores: Datos que se utilizarán para que el perceptrón multicapa analice
		/// </summary>
		private static readonly string[] NombresColores = { "Rojo", "Verde", "Azul", "Amarillo", "Rosa", "Naranja", "Morado", "Cyan", "Gris", "Café", "Negro", "Blanco" };

		/// <summary>
		/// Arreglo bidimensional el cual obtendrá los datos de entrada
		/// </summary>
		private static readonly double[][] Entradas =
		{
			// Rojo
			new double[] { 255, 0, 0 },       // Rojo puro
			new double[] { 200, 0, 0 },       // Rojo oscuro
			new double[] { 255, 50, 50 },     // Rojo claro
			new double[] { 128, 0, 0 },       // Rojo oscuro
			new double[] { 255, 100, 100 },   // Rojo pastel

			// Verde
			new double[] { 0, 255, 0 },       // Verde puro
			new double[] { 0, 200, 0 },       // Verde oscuro
			new double[] { 50, 255, 50 },     // Verde claro
			new double[] { 0, 128, 0 },       // Verde oscuro
			new double[] { 100, 255, 100 },   // Verde pastel

			// Azul
			new double[] { 0, 0, 255 },       // Azul puro
			new double[] { 0, 0, 200 },       // Azul oscuro
			new double[] { 50, 50, 255 },     // Azul claro
			new double[] { 0, 0, 128 },       // Azul oscuro
			new double[] { 100, 100, 255 },   // Azul pastel

			// Amarillo
			new double[] { 255, 255, 0 },     // Amarillo puro
			new double[] { 200, 200, 0 },     // Amarillo oscuro
			new double[] { 255, 255, 100 },   // Amarillo claro
			new double[] { 128, 128, 0 },     // Amarillo oscuro
			new double[] { 255, 255, 150 },   // Amarillo pastel

			// Rosa
			new double[] { 255, 192, 203 },   // Rosa puro
			new double[] { 255, 182, 193 },   // Rosa claro
			new double[] { 255, 105, 180 },   // Rosa fuerte
			new double[] { 255, 160, 122 },   // Rosa salmón
			new double[] { 255, 20, 147 },    // Rosa profundo

			// Naranja
			new double[] { 255, 165, 0 },     // Naranja puro
			new double[] { 255, 195, 77 },    // Naranja claro
			new double[] { 255, 120, 0 },     // Naranja oscuro
			new double[] { 255, 200, 100 },   // Naranja pastel
			new double[] { 255, 150, 50 },    // Naranja medio

			// Morado
			new double[] { 128, 0, 128 },     // Morado puro
			new double[] { 102, 0, 102 },     // Morado oscuro
			new double[] { 153, 50, 204 },    // Morado claro (Orquídea)
			new double[] { 75, 0, 130 },      // Morado oscuro (Índigo)
			new double[] { 147, 112, 219 },   // Morado pastel (Medio)

			// Cyan
			new double[] { 0, 255, 255 },     // Cyan puro
			new double[] { 0, 200, 200 },     // Cyan oscuro
			new double[] { 0, 128, 128 },     // Cyan oscuro (Verde azulado)
			new double[] { 64, 224, 208 },    // Cyan claro (Turquesa)
			new double[] { 72, 209, 204 },    // Cyan medio (Turquesa medio)

			// Gris
			new double[] { 128, 128, 128 },   // Gris medio
			new double[] { 105, 105, 105 },   // Gris oscuro (DimGray)
			new double[] { 169, 169, 169 },   // Gris claro (DarkGray)
			new double[] { 96, 96, 96 },      // Gris muy oscuro
			new double[] { 64, 64, 64 },      // Gris casi negro

			// Café claro (Café con leche)
			new double[] { 210, 180, 140 },   // Café con leche claro
			new double[] { 205, 170, 125 },   // Café con leche medio
			new double[] { 200, 160, 110 },   // Café con leche oscuro
											 
			// Café medio (Café americano)   
			new double[] { 139, 69, 19 },     // Café americano claro
			new double[] { 120, 60, 15 },     // Café americano medio
			new double[] { 100, 50, 10 },     // Café americano oscuro
											 
			// Café oscuro (Espresso)        
			new double[] { 80, 40, 10 },      // Espresso claro
			new double[] { 70, 35, 8 },       // Espresso medio
			new double[] { 60, 30, 5 },       // Espresso oscuro
			new double[] { 50, 25, 5 },       // Espresso muy oscuro
											 
			// Negro                         
			new double[] { 0, 0, 0 },         // Negro puro
			new double[] { 10, 10, 10 },      // Negro casi puro
			new double[] { 20, 20, 20 },      // Negro oscuro
			new double[] { 30, 30, 30 },      // Negro grisáceo
			new double[] { 40, 40, 40 },      // Negro grisáceo claro
											 
			// Blanco                        
			new double[] { 255, 255, 255 },   // Blanco puro
			new double[] { 245, 245, 245 },   // Blanco humo (Smoke)
			new double[] { 250, 250, 250 },   // Blanco nieve (Snow)
			new double[] { 240, 240, 240 },   // Blanco claro (LightWhite)
			new double[] { 230, 230, 230 }    // Blanco grisáceo

		};

		/// <summary>
		/// Salidas esperadas que sean y se aproximen a la salida obtenida del percetprón.
		/// </summary>
		private static readonly double[][] Salidas =
		{
			// Rojo
			new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // Rojo puro
			new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // Rojo oscuro
			new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // Rojo claro
			new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // Rojo oscuro
			new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // Rojo pastel
														  
			// Verde                                      
			new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // Verde puro
			new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // Verde oscuro
			new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // Verde claro
			new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // Verde oscuro
			new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // Verde pastel
														  
			// Azul                                       
			new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // Azul puro
			new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // Azul oscuro
			new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // Azul claro
			new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // Azul oscuro
			new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // Azul pastel
														  
			// Amarillo                                   
			new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 }, // Amarillo puro
			new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 }, // Amarillo oscuro
			new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 }, // Amarillo claro
			new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 }, // Amarillo oscuro
			new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 }, // Amarillo pastel
														  
			// Rosa                                       
			new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 }, // Rosa claro
			new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 }, // Rosa puro
			new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 }, // Rosa fuerte
			new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 }, // Rosa salmón
			new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 }, // Rosa profundo
														  
			// Naranja                                    
			new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }, // Naranja puro
			new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }, // Naranja claro
			new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }, // Naranja oscuro
			new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }, // Naranja pastel
			new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }, // Naranja medio
														  
			// Morado                                     
			new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }, // Rojo puro
			new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }, // Rojo oscuro
			new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }, // Rojo claro
			new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }, // Rojo oscuro
			new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }, // Rojo pastel
														  
			// Cyan                                       
			new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 }, // Rojo puro
			new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 }, // Rojo oscuro
			new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 }, // Rojo claro
			new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 }, // Rojo oscuro
			new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 }, // Rojo pastel
														  
			// Gris                                       
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 }, // Gris medio
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 }, // Gris oscuro (DimGray)
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 }, // Gris claro (DarkGray)
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 }, // Gris muy oscuro
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 }, // Gris casi negro

			// Café claro (Café con leche)
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }, // Café con leche claro
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }, // Café con leche medio
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }, // Café con leche oscuro
													  
			// Café medio (Café americano)            
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }, // Café americano claro
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }, // Café americano medio
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }, // Café americano oscuro
														
			// Café oscuro (Espresso)                 
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }, // Espresso claro
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }, // Espresso medio
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }, // Espresso oscuro
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }, // Espresso muy oscuro

			// Negro
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 }, // Negro puro
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 }, // Negro casi puro
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 }, // Negro oscuro
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 }, // Negro grisáceo
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 }, // Negro grisáceo claro

			// Blanco
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, // Blanco puro
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, // Blanco humo (Smoke)
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, // Blanco nieve (Snow)
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, // Blanco claro (LightWhite)
			new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }  // Blanco grisáceo

		};

		/// <summary>
		/// Constructor para iniciar y comprobar si existe la carpeta "Archivos". Si no la encuentra la crea.
		/// </summary>
		public VariablesGlobales()
		{
			try
			{
				if (!Directory.Exists(Ruta))
				{
					MessageBox.Show("Se ha creado el directorio.", "Archivos");
					Directory.CreateDirectory(Ruta);
				}
			}
			catch(Exception e)
			{
				Console.WriteLine($"Error al crear el directorio: {e.Message}");
			}
		}

		/// <summary>
		/// Método para regresar el nombre del archivo de configuración.
		/// </summary>
		/// <returns>Regresa el nombre del archivo.</returns>
		public string GetArchivoConfiguracion()
		{
			return Configuracion;
		}

		/// <summary>
		/// Método para regresar el nombre del archivo de datos.
		/// </summary>
		/// <returns>Regresa el nombre del archivo.</returns>
		public string GetArchivoDatos()
		{
			return Datos;
		}

		/// <summary>
		/// Método para regresar el error mínimo.
		/// </summary>
		/// <returns>Regresa el error mínimo.</returns>
		public double GetErrorMinimo()
		{
			return ErrorMinimo;
		}
		
		/// <summary>
		/// Método para regresar el formato de los archivos.
		/// </summary>
		/// <returns>Regresa el formate de los archivos.</returns>
		public string GetFormato()
		{
			return FormatoArchivos;
		}

		/// <summary>
		/// Método para regresar la ruta en la cual guardar los archivos.
		/// </summary>
		/// <returns>Regresa la ruta.</returns>
		public string GetRuta()
		{
			return Ruta;
		}

		/// <summary>
		/// Método para regresar los datos a elección del PML.
		/// </summary>
		/// <returns>Regresa las opciones que tiene el perceptron.</returns>
		public string[] GetColores()
		{
			return NombresColores;
		}

		/// <summary>
		/// Método para regresar los datos del perceptron.
		/// </summary>
		/// <returns>Regresa los datos del perceptron.</returns>
		public int[] GetDatosPerceptron()
		{
			return DatosPerceptron;
		}

		/// <summary>
		/// Método para regresar la Tasa de aprendizaje.
		/// </summary>
		/// <returns>Regresa la Tasa de aprendizaje.</returns>
		public double GetTasaAprendizaje()
		{
			return TasaAprendizaje;
		}

		/// <summary>
		/// Método para regresar las épocas de entrenamiento.
		/// </summary>
		/// <returns>Regresa las épocas.</returns>
		public int GetEpocas()
		{
			return Epocas;
		}

		/// <summary>
		/// Método para regresar el valor mínimo.
		/// </summary>
		/// <returns>Regresa el mínimo.</returns>
		public int GetMin()
		{
			return Min;
		}

		/// <summary>
		/// Método para regresar el valor máximo.
		/// </summary>
		/// <returns>Regresa el máximo.</returns>
		public int GetMax()
		{
			return Max;
		}

		/// <summary>
		/// Método para regresar las entradas para el PML.
		/// </summary>
		/// <returns>Regresa las entradas.</returns>
		public double[][] GetEntradas()
		{
			return Entradas;
		}

		
		/// <summary>
		/// Método para regresar las salidas esperadas del PML.
		/// </summary>
		/// <returns>Regresa las salidas.</returns>
		public double[][] GetSalidas()
		{
			return Salidas;
		}
	}
}
