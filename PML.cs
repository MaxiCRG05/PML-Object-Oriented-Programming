using System;
using System.Drawing;
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
		public readonly Archivos archivo = new Archivos(VariablesGlobales.Ruta);

		/// <summary>
		/// Constructor de la clase PML
		/// </summary>
		/// <param name="n">Arreglo que define el número de neuronas en cada capa.</param>
		public PML(int[] n)
		{
			Capas = new Capa[n.Length];
			for (int c = 0; c < n.Length; c++)
			{
				int neuronasCapaSiguiente = (c == n.Length - 1) ? 0 : n[c + 1];
				Capa.TipoCapa tipo = (c == 0) ? Capa.TipoCapa.Entrada : (c == n.Length - 1) ? Capa.TipoCapa.Salida : Capa.TipoCapa.Oculta;
				Capas[c] = new Capa(n[c], neuronasCapaSiguiente, tipo);
			}
		}

		/// <summary>
		/// Método para entrenar la red neuronal
		/// </summary>
		public void Entrenar()
		{
			double mejorError = double.MaxValue;
			int epocasSinMejora = 0;

			for (int epoca = 0; epoca < VariablesGlobales.Epocas; epoca++)
			{
				double errorEpoca = 0;

				for (int i = 0; i < VariablesGlobales.Entradas.Length; i++)
				{
					double[] salidaObtenida = Propagacion(VariablesGlobales.Entradas[i]);
					Retropropagacion(VariablesGlobales.Salidas[i]);

					for (int j = 0; j < VariablesGlobales.Salidas[i].Length; j++)
					{
						errorEpoca += Math.Pow(VariablesGlobales.Salidas[i][j] - salidaObtenida[j], 2);
					}
				}

				errorEpoca /= (VariablesGlobales.Entradas.Length * VariablesGlobales.Salidas[0].Length);
				Console.WriteLine($"Epoca {epoca} Error: {errorEpoca}");

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
						Console.WriteLine($"Entrenamiento detenido en la época {epoca + 1}. Error: {errorEpoca}.");
						MessageBox.Show($"Entrenamiento detenido en la época {epoca + 1}. Error: {errorEpoca}.", "Entrenamiento");
						break;
					}
				}

				if (errorEpoca <= VariablesGlobales.ErrorMinimo)
				{
					Console.WriteLine($"Entrenamiento detenido en la época {epoca + 1}. El error se disminuyó: {errorEpoca}.");
					MessageBox.Show($"Entrenamiento detenido en la época {epoca + 1}. El error se disminuyó: {errorEpoca}.", "Entrenamiento");
					break;
				}
			}
			MessageBox.Show($"Entrenamiento finalizado.", "Entrenamiento");
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
				for (int j = 0; j < Capas[c].Neuronas.Length; j++)
				{
					double suma = 0;
					for (int k = 0; k < Capas[c - 1].Neuronas.Length; k++)
					{
						suma += Capas[c - 1].Neuronas[k].Salida * Capas[c].Neuronas[j].Pesos[k];
					}
					Capas[c].Neuronas[j].Salida = FuncionActivacion(suma + Capas[c].Neuronas[j].Bias);
				}
			}

			double[] salidas = new double[Capas[Capas.Length - 1].Neuronas.Length];
			for (int i = 0; i < salidas.Length; i++)
			{
				salidas[i] = Capas[Capas.Length - 1].Neuronas[i].Salida;
			}

			if (Capas[Capas.Length - 1].Tipo == Capa.TipoCapa.Salida)
			{
				return Softmax(salidas);
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
			for (int i = 0; i < Capas[Capas.Length - 1].Neuronas.Length; i++)
			{
				double output = Capas[Capas.Length - 1].Neuronas[i].Salida;
				double error = salidaEsperada[i] - output;
				Capas[Capas.Length - 1].Neuronas[i].Delta = error * FuncionDeActivacionDerivada(output);
			}

			for (int c = Capas.Length - 2; c >= 0; c--)
			{
				for (int j = 0; j < Capas[c].Neuronas.Length; j++)
				{
					double error = 0;
					for (int i = 0; i < Capas[c + 1].Neuronas.Length; i++)
					{
						error += Capas[c + 1].Neuronas[i].Delta * Capas[c + 1].Neuronas[i].Pesos[j];
					}
					Capas[c].Neuronas[j].Delta = error * FuncionDeActivacionDerivada(Capas[c].Neuronas[j].Salida);
				}
			}

			for (int c = 0; c < Capas.Length - 1; c++)
			{
				for (int j = 0; j < Capas[c + 1].Neuronas.Length; j++)
				{
					for (int i = 0; i < Capas[c].Neuronas.Length; i++)
					{
						Capas[c + 1].Neuronas[j].Pesos[i] += VariablesGlobales.TasaAprendizaje * Capas[c + 1].Neuronas[j].Delta * Capas[c].Neuronas[i].Salida;
					}
					Capas[c + 1].Neuronas[j].Bias += VariablesGlobales.TasaAprendizaje * Capas[c + 1].Neuronas[j].Delta;
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
				for(int c = 0; c < Capas.Length; c++)
				{
					archivo.EscribirArchivo($"Capa {c}:", VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);

					for (int j = 0; j < Capas[c].Neuronas.Length ; j++)
					{
						archivo.EscribirArchivo($"Neurona {j}:", VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);

						for(int p = 0; p < Capas[c].Neuronas[j].Pesos.Length ; p++)
						{
							archivo.EscribirArchivo($"Peso = {Capas[c].Neuronas[j].Pesos[p]}", VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);
						}

						archivo.EscribirArchivo($"Bias = {Capas[c].Neuronas[j].Bias}", VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);
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