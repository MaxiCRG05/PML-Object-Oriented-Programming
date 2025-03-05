using System;
using System.Windows.Forms;

namespace Perceptron_Multicapa_Colores
{
	/// <summary>
	/// Clase del Perceptrón MultiCapa, se encarga del entrenamiento y la propagación.
	/// </summary>
	class PML
	{
		/// <summary>
		/// archivoPesos: Variable para poder obtener el nombre del archivo de configuración.
		/// formatoArchivos: Variable para poder obtener el formato de los archivos.
		/// </summary>
		private readonly string archivoPesos, formatoArchivos;

		/// <summary>
		/// errorMinimo: Variable para obtener el error mínimo.
		/// </summary>
		private readonly double errorMinimo;

		/// <summary>
		/// Capas: Arreglo de capas.
		/// </summary>
		private readonly Capa[] Capas;

		/// <summary>
		/// Instancia de la clase VariablesGlobales para definir los parámetros que se utilizarán para entrenar la red.
		/// Entre otras cosas.
		/// </summary>
		private readonly VariablesGlobales variables = new VariablesGlobales();

		/// <summary>
		/// Instancia de la clase Archivos para manejar los archivos de guardado de la red neuronal.
		/// </summary>
		public readonly Archivos archivo;

		/// <summary>
		/// Método constructor de la clase PML, aquí se inicializan todas las cosas que se utilizarán.
		/// </summary>
		/// <param name="layers">Capas obtenidas de las variables cuando se instancie un PML.</param>
		public PML(int[] layers)
		{
			formatoArchivos = variables.GetFormato();
			archivoPesos = variables.GetArchivoConfiguracion();
			errorMinimo = variables.GetErrorMinimo();
			archivo = new Archivos(variables.GetRuta());

			Capas = new Capa[layers.Length];

			for (int i = 0; i < layers.Length; i++)
			{
				int neuronasCapaSiguiente = (i == 0) ? 0 : layers[i - 1]; 
				Capa.TipoCapa tipo;

				if (i == 0)
				{
					tipo = Capa.TipoCapa.Entrada;
				}
				else if (i == layers.Length - 1)
				{
					tipo = Capa.TipoCapa.Salida;
				}
				else
				{
					tipo = Capa.TipoCapa.Oculta;
				}

				Capas[i] = new Capa(layers[i], neuronasCapaSiguiente, tipo); 
			}
		}


		/// <summary>
		/// Método para entrenar la red neuronal.
		/// </summary>
		/// <param name="entradas">Entradas que se procesarán para el entrenamiento de la red neuronal.</param>
		/// <param name="salidas">Salidas esperadas para el entrenamiento de la red neuronal.</param>
		/// <param name="tasaAprendizaje">Tasa de aprendizaje de la red neuronal.</param>
		/// <param name="epocas">Epocas que realizará el entrenamiento.</param>
		/// <param name="min">Valor mínimo con base al contexto de lo que se quiera entrenar.Normalización.</param>
		/// <param name="max">Valor máximo con base al contexto de lo que se quiera entrenar.Normalización.</param>
		public void Entrenar(double[][] entradas, double[][] salidas, double tasaAprendizaje, int epocas, int min, int max)
		{

			for (int epoca = 0; epoca < epocas; epoca++)
			{
				double errorEpoca = 0;

				for (int e = 0; e < entradas.Length; e++)
				{
					double[] entradaNormalizada = NormalizarDatos(entradas[e], min, max);
					double[] salidaRed = Propagacion(entradaNormalizada, min, max);

					for (int s = 0; s < salidas[e].Length; s++)
					{
						errorEpoca += Math.Pow(salidas[e][s] - salidaRed[s], 2);
					}

					double[] errores = new double[salidas[e].Length];

					for (int s = 0; s < salidas[e].Length; s++)
					{
						errores[s] = salidas[e][s] - salidaRed[s];
					}

					Retropropagacion(errores, tasaAprendizaje, entradaNormalizada);
				}

				errorEpoca /= (entradas.Length * salidas[0].Length);

				Console.WriteLine($"Epoca: {epoca}\tError: {errorEpoca}");

				if (errorEpoca <= errorMinimo)
				{
					MessageBox.Show($"Entrenamiento detenido en la época {epoca + 1}. El error se disminuyó: {errorEpoca}", "PML");
					break;
				}
			}

			MessageBox.Show("Entrenamiento finalizado.", "PML");
		}

		public double[] Propagacion(double[] entradas, int min, int max)
		{
			double[] salidas = NormalizarDatos(entradas, min, max);
			for (int c = 0; c < Capas.Length; c++)
			{
				salidas = Capas[c].Propagacion(salidas);
			}
			return salidas;
		}

		private void Retropropagacion(double[] errores, double tasaAprendizaje, double[] entradas)
		{
			for (int c = Capas.Length - 1; c >= 0; c--)
			{
				Capas[c].Retropropagacion(errores, tasaAprendizaje, entradas);
			}
		}

		public double[] NormalizarDatos(double[] entradas, int min, int max)
		{
			double[] resultado = new double[entradas.Length];
			for (int i = 0; i < entradas.Length; i++)
			{
				resultado[i] = (entradas[i] - min) / (max - min);
			}
			return resultado;
		}

		public bool CargarDatos()
		{
			try
			{
				string line;
				int capaActual = -1;
				int neuronaActual = 0;
				int pesoActual = 0;

				while ((line = archivo.LeerArchivo(archivoPesos + formatoArchivos)) != null)
				{
					if (line.StartsWith("Capa"))
					{
						capaActual++;
						neuronaActual = 0;
						pesoActual = 0;
					}
					else if (line.StartsWith("Peso"))
					{
						if (Capas[capaActual].Tipo != Capa.TipoCapa.Entrada) // Solo cargar pesos si no es capa de entrada
						{
							string[] partes = line.Split('=');
							double peso = double.Parse(partes[1].Trim());

							if (capaActual >= 0 && capaActual < Capas.Length && neuronaActual < Capas[capaActual].Neuronas.Length)
							{
								Capas[capaActual].Neuronas[neuronaActual].Pesos[pesoActual] = peso;
								pesoActual++;
								if (pesoActual >= Capas[capaActual].Neuronas[neuronaActual].Pesos.Length)
								{
									pesoActual = 0;
									neuronaActual++;
								}
							}
						}
					}
					else if (line.StartsWith("Sesgo"))
					{
						if (Capas[capaActual].Tipo != Capa.TipoCapa.Entrada) // Solo cargar sesgos si no es capa de entrada
						{
							string[] partes = line.Split('=');
							double sesgo = double.Parse(partes[1].Trim());

							if (capaActual >= 0 && capaActual < Capas.Length && neuronaActual < Capas[capaActual].Neuronas.Length)
							{
								Capas[capaActual].Neuronas[neuronaActual].Bias = sesgo;
								neuronaActual++;
							}
						}
					}
				}

				MessageBox.Show("Configuración cargada correctamente.", "Perceptron");
				return true;
			}
			catch (Exception e)
			{
				MessageBox.Show($"No se ha podido cargar la configuración.\nError:\n {e.Message}\nStackTrace:\n{e.StackTrace}");
				return false;
			}
		}

		public void GuardarDatos()
		{
			try
			{
				for (int i = 0; i < Capas.Length; i++)
				{
					if (Capas[i].Tipo != Capa.TipoCapa.Entrada) // Solo guardar si no es capa de entrada
					{
						archivo.EscribirArchivo($"Capa {i}:", archivoPesos + formatoArchivos);
						for (int j = 0; j < Capas[i].Neuronas.Length; j++)
						{
							for (int k = 0; k < Capas[i].Neuronas[j].Pesos.Length; k++)
							{
								archivo.EscribirArchivo($"Peso[{i}][{j}][{k}] = {Capas[i].Neuronas[j].Pesos[k]}", archivoPesos + formatoArchivos);
							}
							archivo.EscribirArchivo($"Sesgo[{i}][{j}] = {Capas[i].Neuronas[j].Bias}", archivoPesos + formatoArchivos);
						}
					}
				}

				MessageBox.Show("Se ha guardado la configuración correctamente.", "Perceptron");
			}
			catch (Exception e)
			{
				MessageBox.Show($"No se ha podido guardar la configuración.\nError:\n {e.Message}");
			}
		}
	}
}