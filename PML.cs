using System;
using System.Windows.Forms;

namespace Perceptron_Multicapa_Colores
{
	/// <summary>
	/// Clase del Perceptrón MultiCapa
	/// </summary>
	class PML
	{
		private readonly string archivoPesos, formatoArchivos;
		private readonly double errorMinimo;
		private readonly Capa[] Capas;
		private readonly VariablesGlobales variables = new VariablesGlobales();
		public readonly Archivos archivo;

		public PML(int[] layers)
		{
			formatoArchivos = variables.GetFormato();
			archivoPesos = variables.GetArchivoConfiguracion();
			errorMinimo = variables.GetErrorMinimo();
			archivo = new Archivos(variables.GetRuta());

			Capas = new Capa[layers.Length];

			for (int i = 0; i < layers.Length; i++)
			{
				int neuronasCapaAnterior = (i == 0) ? 0 : layers[i - 1]; 
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

				Capas[i] = new Capa(layers[i], neuronasCapaAnterior, tipo); 
			}
		}

		public void Entrenar(double[][] entradas, double[][] salidas, double tasaAprendizaje, int epocas, int min, int max)
		{
			for (int epoca = 0; epoca < epocas; epoca++)
			{
				double errorEpoca = 0;

				for (int e = 0; e < entradas.Length; e++)
				{
					double[] entradaNormalizada = NormalizarEntradas(entradas[e], min, max);
					double[] salidaRed = Propagacion(entradaNormalizada, min, max);

					for (int i = 0; i < salidas[e].Length; i++)
					{
						errorEpoca += Math.Pow(salidas[e][i] - salidaRed[i], 2);
					}

					double[] errores = new double[salidas[e].Length];
					for (int i = 0; i < salidas[e].Length; i++)
					{
						errores[i] = salidas[e][i] - salidaRed[i];
					}
					Retropropagacion(errores, tasaAprendizaje, entradaNormalizada);
				}

				errorEpoca /= (entradas.Length * salidas[0].Length);

				Console.WriteLine($"Época {epoca + 1}, Error: {errorEpoca}");

				if (double.IsNaN(errorEpoca))
				{
					Console.WriteLine($"Error en el entrenamiento");
					break;
				}
				else if (errorEpoca <= errorMinimo)
				{
					Console.WriteLine($"Entrenamiento detenido en la época {epoca + 1}. Error mínimo alcanzado: {errorEpoca}");
					break;
				}
			}
		}

		public double[] Propagacion(double[] entradas, int min, int max)
		{
			double[] salidas = NormalizarEntradas(entradas, min, max);
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

		public double NormalizarSalidas(double x, int min, int max)
		{
			return (x - min) / (max - min);
		}

		public double[] NormalizarEntradas(double[] entradas, int min, int max)
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