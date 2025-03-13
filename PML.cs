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
		/// <summary>
		/// Arreglo de capas de la red neuronal.
		/// </summary>
		private readonly Capa[] Capas;

		/// <summary>
		/// Archivo para manejar los archivos.
		/// </summary>
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
				Capa capaSiguiente = (c < layers.Length - 1) ? Capas[c + 1] : null;
				Capas[c] = new Capa(layers[c], neuronasCapaSiguiente, tipo, layers, capaSiguiente);
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
		public void Entrenar()
		{
			double mejorError = double.MaxValue;
			int epocasSinMejora = 0;

			for (int epoca = 0; epoca < VariablesGlobales.Epocas; epoca++)
			{
				double errorEpoca = 0;

				for (int i = 0; i < VariablesGlobales.Entradas.Length; i++)
				{
					double[] salidaCalculada = Propagacion(VariablesGlobales.Entradas[i]);
					Retropropagacion(VariablesGlobales.Salidas[i]);

					for (int j = 0; j < VariablesGlobales.Salidas[i].Length; j++)
					{
						errorEpoca += Math.Pow(VariablesGlobales.Salidas[i][j] - salidaCalculada[j], 2);
					}
				}

				errorEpoca /= (VariablesGlobales.Entradas.Length * VariablesGlobales.Salidas[0].Length);

				if (errorEpoca < mejorError)
				{
					mejorError = errorEpoca;
					epocasSinMejora = 0;
				}
				else
				{
					epocasSinMejora++;
					if (epocasSinMejora >= VariablesGlobales.Paciencia)
					{
						Console.WriteLine($"Entrenamiento detenido en la época {epoca + 1}. Error: {errorEpoca}");
						break;
					}
				}

				if (errorEpoca <= VariablesGlobales.ErrorMinimo)
				{
					Console.WriteLine($"Entrenamiento detenido en la época {epoca + 1}. El error se disminuyó: {errorEpoca}");
					MessageBox.Show($"Entrenamiento detenido en la época {epoca + 1}. El error se disminuyó: {errorEpoca}", "Entrenamiento");
					break;
				}
			}
		}

		/// <summary>
		/// Realiza la propagación hacia adelante
		/// </summary>
		/// <param name="entradas">Entradas de la red.</param>
		/// <returns>Salida de la red.</returns>
		public double[] Propagacion(double[] entradas)
		{
			entradas = NormalizarDatos(entradas);

			for (int i = 0; i < entradas.Length; i++)
			{
				Capas[0].Neuronas[i].Salida = entradas[i];
			}

			for (int c = 1; c < Capas.Length; c++)
			{
				for (int i = 0; i < Capas[c].Neuronas.Length; i++)
				{
					double suma = 0;
					for (int j = 0; j < Capas[c - 1].Neuronas.Length; j++)
					{
						suma += Capas[c - 1].Neuronas[j].Salida * Capas[c - 1].Neuronas[j].Pesos[i];
					}
					suma += Capas[c].Neuronas[i].Bias;
					Capas[c].Neuronas[i].Salida = FuncionActivacion(suma);
				}
			}

			// Aplicar Softmax en la capa de salida si es necesario
			if (Capas[Capas.Length - 1].Tipo == Capa.TipoCapa.Salida)
			{
				double[] salidas = new double[Capas[Capas.Length - 1].Neuronas.Length];
				for (int i = 0; i < salidas.Length; i++)
				{
					salidas[i] = Capas[Capas.Length - 1].Neuronas[i].Salida;
				}
				return Softmax(salidas);
			}

			// Devolver las salidas de la última capa
			double[] salidasFinales = new double[Capas[Capas.Length - 1].Neuronas.Length];
			for (int i = 0; i < salidasFinales.Length; i++)
			{
				salidasFinales[i] = Capas[Capas.Length - 1].Neuronas[i].Salida;
			}
			return salidasFinales;
		}

		/// <summary>
		/// Realiza la retropropagación
		/// </summary>
		/// <param name="salidaEsperada">Salida esperada.</param>
		/// <param name="tasaAprendizaje">Tasa de aprendizaje.</param>
		private void Retropropagacion(double[] salidaEsperada)
		{
			// Calcular el error en la capa de salida
			for (int i = 0; i < Capas[Capas.Length - 1].Neuronas.Length; i++)
			{
				double output = Capas[Capas.Length - 1].Neuronas[i].Salida;
				double error = salidaEsperada[i] - output;
				Capas[Capas.Length - 1].Neuronas[i].Delta = error * FuncionDeActivacionDerivada(output);
			}

			// Retropropagación del error
			for (int c = Capas.Length - 2; c >= 0; c--)
			{
				for (int i = 0; i < Capas[c].Neuronas.Length; i++)
				{
					double error = 0;
					for (int j = 0; j < Capas[c + 1].Neuronas.Length; j++)
					{
						error += Capas[c + 1].Neuronas[j].Delta * Capas[c].Neuronas[i].Pesos[j];
					}
					Capas[c].Neuronas[i].Delta = error * FuncionDeActivacionDerivada(Capas[c].Neuronas[i].Salida);
				}
			}

			// Actualización de pesos y sesgos
			for (int c = 0; c < Capas.Length - 1; c++)
			{
				for (int i = 0; i < Capas[c].Neuronas.Length; i++)
				{
					for (int j = 0; j < Capas[c + 1].Neuronas.Length; j++)
					{
						Capas[c].Neuronas[i].Pesos[j] += VariablesGlobales.TasaAprendizaje * Capas[c + 1].Neuronas[j].Delta * Capas[c].Neuronas[i].Salida;
					}
					Capas[c].Neuronas[i].Bias += VariablesGlobales.TasaAprendizaje * Capas[c + 1].Neuronas[i].Delta;
				}
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
		/// Realiza la predicción de la red.
		/// </summary>
		/// <param name="x">Datos que se van a procesar</param>
		/// <returns>Los datos ya procesados</returns>
		private double[] Softmax(double[] x)
		{
			double[] exponenciales = new double[x.Length];
			double sumaExponenciales = 0;

			for (int i = 0; i < x.Length; i++)
			{
				exponenciales[i] = Math.Exp(x[i]);
				sumaExponenciales += exponenciales[i];
			}

			for (int i = 0; i < x.Length; i++)
			{
				exponenciales[i] /= sumaExponenciales;
			}

			return exponenciales;
		}

		/// <summary>
		/// Método para la función de activación: FUNCION RELU, LEAKY RELU Y SIGMOIDE
		/// </summary>
		/// <param name="x">Valor de entrada.</param>
		/// <returns>Regresa el valor de entrada procesada con base a la función de activación.</returns>
		private double FuncionActivacion(double x)
		{
			//SIGMOIDE
			//return 1 / (1 + Math.Exp(-x));

			//RELU
			//return Math.Max(0,x);

			// Leaky ReLU
			return x > 0 ? x : 0.01 * x;

			//ELU
			//return x > 0 ? x : 0.01 * (Math.Exp(x) - 1);
		}

		/// <summary>
		/// Método para la función de activación derivada: FUNCION RELU, LEAKY RELU Y SIGMOIDE DERIVADA
		/// </summary>
		/// <param name="x">Valor de entrada.</param>
		/// <returns>La entrada ya procesada por la función de activación derivada.</returns>
		private double FuncionDeActivacionDerivada(double x)
		{
			//SIGMOIDE
			//return x * (1 - x);

			//RELU
			//return x > 0 ? 1 : 0;

			//Leaky ReLU
			return x > 0 ? 1 : 0.01;

			//ELU
			//return x > 0 ? 1 : 0.01 * Math.Exp(x);
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