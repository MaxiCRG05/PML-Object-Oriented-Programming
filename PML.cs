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
		private readonly Capa[] Capas;
		public readonly Archivos archivo;

		/// <summary>
		/// Constructor de la clase PML
		/// </summary>
		/// <param name="layers">Arreglo que define el número de neuronas en cada capa.</param>
		public PML(int[] layers)
		{
			Capas = new Capa[layers.Length];
			for (int c = 0; c < layers.Length; c++)
			{
				int neuronasCapaSiguiente = (c == layers.Length - 1) ? 0 : layers[c + 1];
				Capa.TipoCapa tipo = (c == 0) ? Capa.TipoCapa.Entrada : (c == layers.Length - 1) ? Capa.TipoCapa.Salida : Capa.TipoCapa.Oculta;
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
		public void Entrenar(double[][] entradas, double[][] salidas, int epocas)
		{
			for (int epoca = 0; epoca < epocas; epoca++)
			{
				double errorEpoca = 0;

				for (int e = 0; e < entradas.Length; e++)
				{
					double[] salidaRed = Propagacion(entradas[e]);

					Retropropagacion(salidas[e]);

					for (int s = 0; s < salidas[e].Length; s++)
					{
						errorEpoca += Math.Pow(salidas[e][s] - salidaRed[s], 2);
					}
				}

				errorEpoca /= (entradas.Length * salidas[0].Length);

				Console.WriteLine($"Época: {epoca + 1}, Error: {errorEpoca}");
			}
		}

		/// <summary>
		/// Realiza la propagación hacia adelante
		/// </summary>
		/// <param name="entradas">Entradas de la red.</param>
		/// <returns>Salida de la red.</returns>
		public double[] Propagacion(double[] entradas)
		{
			double[] salidas = NormalizarDatos(entradas);

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
		private void Retropropagacion(double[] salidaEsperada)
		{
			for (int c = Capas.Length - 1; c >= 0; c--)
			{
				Capas[c].Retropropagacion(salidaEsperada);
			}
		}

		/// <summary>
		/// Normaliza un valor individual
		/// </summary>
		/// <param name="entrada">Valor a normalizar.</param>
		/// <returns>Valor normalizado.</returns>
		public double NormalizarDatos(double entrada)
		{
			return (entrada - VariablesGlobales.Min) / (VariablesGlobales.Max - VariablesGlobales.Min);
		}

		/// <summary>
		/// Normaliza un arreglo de valores
		/// </summary>
		/// <param name="entradas">Arreglo de valores a normalizar.</param>
		/// <returns>Arreglo de valores normalizados.</returns>
		public double[] NormalizarDatos(double[] entradas)
		{
			double[] resultado = new double[entradas.Length];
			for (int i = 0; i < entradas.Length; i++)
			{
				resultado[i] = NormalizarDatos(entradas[i]);
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

				while ((line = archivo.LeerArchivo(VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos)) != null)
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
						archivo.EscribirArchivo($"Capa {i}:", VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);
						for (int j = 0; j < Capas[i].Neuronas.Length; j++)
						{
							for (int k = 0; k < Capas[i].Neuronas[j].Pesos.Length; k++)
							{
								archivo.EscribirArchivo($"Peso[{i}][{j}][{k}] = {Capas[i].Neuronas[j].Pesos[k]}", VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);
							}
							archivo.EscribirArchivo($"Sesgo[{i}][{j}] = {Capas[i].Neuronas[j].Bias}", VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);
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