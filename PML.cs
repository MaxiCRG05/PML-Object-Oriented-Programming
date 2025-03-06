using System;
using System.Linq;
using System.Windows.Forms;

namespace Perceptron_Multicapa_Colores
{
	/// <summary>
	/// Clase del Perceptrón MultiCapa (PML)
	/// </summary>
	class PML
	{
		private readonly string archivoPesos, formatoArchivos;
		private readonly double errorMinimo;
		private readonly Capa[] Capas;
		private readonly VariablesGlobales variables = new VariablesGlobales();
		public readonly Archivos archivo;

		/// <summary>
		/// Constructor de la clase PML
		/// </summary>
		/// <param name="layers">Arreglo que define el número de neuronas en cada capa.</param>
		public PML(int[] layers)
		{
			formatoArchivos = variables.GetFormato();
			archivoPesos = variables.GetArchivoConfiguracion();
			errorMinimo = variables.GetErrorMinimo();
			archivo = new Archivos(variables.GetRuta());

			// Inicializar las capas
			Capas = new Capa[layers.Length];
			for (int c = 0; c < layers.Length; c++)
			{
				int neuronasCapaSiguiente = (c == 0) ? 0 : layers[c - 1];
				Capa.TipoCapa tipo;

				if (c == 0)
				{
					tipo = Capa.TipoCapa.Entrada;
				}
				else if (c == layers.Length - 1)
				{
					tipo = Capa.TipoCapa.Salida;
				}
				else
				{
					tipo = Capa.TipoCapa.Oculta;
				}

				Capas[c] = new Capa(layers[c], neuronasCapaSiguiente, tipo);
			}
		}

		/// <summary>
		/// Método para entrenar la red neuronal
		/// </summary>
		/// <param name="entradas">Conjunto de datos de entrada.</param>
		/// <param name="salidas">Conjunto de datos de salida esperados.</param>
		/// <param name="tasaAprendizaje">Tasa de aprendizaje.</param>
		/// <param name="epocas">Número de épocas de entrenamiento.</param>
		/// <param name="min">Valor mínimo para normalización.</param>
		/// <param name="max">Valor máximo para normalización.</param>
		public void Entrenar(double[][] entradas, double[][] salidas, double tasaAprendizaje, int epocas, int min, int max)
		{
			double mejorError = double.MaxValue;
			int epocasSinMejora = 0;
			const int paciencia = 1000;

			for (int epoca = 0; epoca < epocas; epoca++)
			{
				double errorEpoca = 0;

				for (int e = 0; e < entradas.Length; e++)
				{
					// Propagación hacia adelante
					double[] salidaRed = Propagacion(entradas[e], min, max);

					// Retropropagación
					Retropropagacion(salidas[e], tasaAprendizaje);

					// Cálculo del error
					for (int s = 0; s < salidas[e].Length; s++)
					{
						errorEpoca += Math.Pow(salidas[e][s] - salidaRed[s], 2);
					}
				}

				errorEpoca /= (entradas.Length * salidas[0].Length);

				// Verificar si el error ha mejorado
				if (errorEpoca < mejorError)
				{
					mejorError = errorEpoca;
					epocasSinMejora = 0;
				}
				else
				{
					epocasSinMejora++;
					if (epocasSinMejora >= paciencia)
					{
						Console.WriteLine($"Entrenamiento detenido en la época {epoca + 1}. Error: {errorEpoca}");
						break;
					}
				}

				Console.WriteLine($"Época: {epoca + 1}, Error: {errorEpoca}");

				// Detener el entrenamiento si el error es menor o igual al error mínimo
				if (errorEpoca <= errorMinimo)
				{
					MessageBox.Show($"Entrenamiento detenido en la época {epoca + 1}. Error: {errorEpoca}", "PML");
					break;
				}
			}

			MessageBox.Show("Entrenamiento finalizado.", "PML");
		}

		/// <summary>
		/// Realiza la propagación hacia adelante
		/// </summary>
		/// <param name="entradas">Entradas de la red.</param>
		/// <param name="min">Valor mínimo para normalización.</param>
		/// <param name="max">Valor máximo para normalización.</param>
		/// <returns>Salida de la red.</returns>
		public double[] Propagacion(double[] entradas, int min, int max)
		{
			double[] salidas = NormalizarDatos(entradas, min, max);

			for (int c = 0; c < Capas.Length; c++)
			{
				salidas = Capas[c].Propagacion(salidas);
			}

			return salidas;
		}

		/// <summary>
		/// Realiza la retropropagación
		/// </summary>
		/// <param name="salidaEsperada">Salida esperada.</param>
		/// <param name="tasaAprendizaje">Tasa de aprendizaje.</param>
		private void Retropropagacion(double[] salidaEsperada, double tasaAprendizaje)
		{
			for (int c = Capas.Length - 1; c >= 0; c--)
			{
				Capas[c].Retropropagacion(salidaEsperada, tasaAprendizaje);
			}
		}

		/// <summary>
		/// Normaliza un valor individual
		/// </summary>
		/// <param name="entrada">Valor a normalizar.</param>
		/// <param name="min">Valor mínimo.</param>
		/// <param name="max">Valor máximo.</param>
		/// <returns>Valor normalizado.</returns>
		public double NormalizarDatos(double entrada, int min, int max)
		{
			return (entrada - min) / (max - min);
		}

		/// <summary>
		/// Normaliza un arreglo de valores
		/// </summary>
		/// <param name="entradas">Arreglo de valores a normalizar.</param>
		/// <param name="min">Valor mínimo.</param>
		/// <param name="max">Valor máximo.</param>
		/// <returns>Arreglo de valores normalizados.</returns>
		public double[] NormalizarDatos(double[] entradas, int min, int max)
		{
			double[] resultado = new double[entradas.Length];
			for (int i = 0; i < entradas.Length; i++)
			{
				resultado[i] = NormalizarDatos(entradas[i], min, max);
			}
			return resultado;
		}

		/// <summary>
		/// Carga los datos de configuración desde un archivo
		/// </summary>
		/// <returns>True si la carga fue exitosa, False en caso contrario.</returns>
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
						if (Capas[capaActual].Tipo != Capa.TipoCapa.Entrada)
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
						if (Capas[capaActual].Tipo != Capa.TipoCapa.Entrada)
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

		/// <summary>
		/// Guarda los datos de configuración en un archivo
		/// </summary>
		public void GuardarDatos()
		{
			try
			{
				for (int i = 0; i < Capas.Length; i++)
				{
					if (Capas[i].Tipo != Capa.TipoCapa.Entrada)
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