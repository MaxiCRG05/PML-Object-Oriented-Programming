using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace Perceptron_Multicapa_Colores
{
	/// <summary>
	/// Clase del Perceptrón MultiCapa (PML)
	/// </summary>
	class PML
	{
		double errorMuestra = 0;
		/// <summary>
		/// Arreglo de capas de la red neuronal.
		/// </summary>
		private readonly List<Capa> capas = new List<Capa>();

		/// <summary>
		/// Epocas alcanzadas para el entrenamiento
		/// </summary>
		public int epocasAlcanzadas = 0;

		/// <summary>
		/// errorEpoca: Es el error de cada epoca.
		/// mejorError: Es el mejor error de cada epoca.
		/// totalPatrones: Es el total de patrones que se van a entrenar.
		/// </summary>
		public double errorEpoca = 0, mejorError = 0, totalPatrones = VariablesGlobales.Entradas.Length;

		/// <summary>
		/// Instancia para manejar los archivos.
		/// </summary>
		private readonly Archivos archivo = new Archivos(VariablesGlobales.Ruta);

		/// <summary>
		/// Constructor de la clase PML
		/// </summary>
		/// <param name="n">Arreglo que define el número de neuronas en cada capa.</param>
		public PML(int[] n)
		{
			for (int c = 0; c < n.Length; c++)
			{
				int neuronasCapaSiguiente = (c == n.Length - 1) ? 0 : n[c + 1];
				Capa.TipoCapa tipo = (c == 0) ? Capa.TipoCapa.Entrada :
									  (c == n.Length - 1) ? Capa.TipoCapa.Salida :
									  Capa.TipoCapa.Oculta;
				capas.Add(new Capa(n[c], neuronasCapaSiguiente, tipo));
			}

			crearConexiones();
		}

		/// <summary>
		/// Crea las conexiones entre las neuronas
		/// </summary>
		public void crearConexiones()
		{
			for (int c = 0; c < capas.Count - 1; c++)
			{
				for (int i = 0; i < capas[c].neuronas.Count; i++)
				{
					for (int j = 0; j < capas[c + 1].neuronas.Count; j++)
					{
						capas[c].neuronas[i].neuronasSiguientes.Add(capas[c + 1].neuronas[j]);
						capas[c + 1].neuronas[j].neuronasAnteriores.Add(capas[c].neuronas[i]);
					}
				}
			}
		}

		public void calcularError(int i)
		{
			errorMuestra = 0;
			for (int j = 0; j < VariablesGlobales.Salidas[i].Length; j++)
			{
				errorMuestra += Math.Pow(VariablesGlobales.Salidas[i][j] - capas[capas.Count - 1].neuronas[j].a, 2);
			}
			errorEpoca += errorMuestra / 2f;
		}


		public bool verificarErrorNAN(int epoca)
		{
			bool fallo = false;

			if (double.IsNaN(errorEpoca))
			{
				Console.WriteLine($"NAN: Epoca {epoca + 1}: Error: {errorEpoca}");
				fallo = true;
			}
			else
				Console.WriteLine($"Epoca {epoca + 1}: Error: {errorEpoca}");

			return fallo;
		}

		public bool verificarPaciencia(int epoca, ref int epocasSinMejora)
		{
			bool fallo = false;

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
					fallo = true;
				}
			}

			return fallo;
		}

		public bool verificarErrorMinimo(int epoca)
		{
			bool fallo = false;

			if (errorEpoca <= VariablesGlobales.ErrorMinimo)
			{
				Console.WriteLine($"Entrenamiento detenido en la época {epoca + 1}. El error se disminuyó: {errorEpoca}");
				MessageBox.Show($"Entrenamiento detenido en la época {epoca + 1}. El error se disminuyó: {errorEpoca}", "Entrenamiento");
				fallo = true;
			}

			return fallo;
		}

		/// <summary>
		/// Método para entrenar la red neuronal
		/// </summary>
		public void entrenar()
		{
			int epocasSinMejora = 0;
			mejorError = double.MaxValue;

			for (int epoca = 0; epoca < VariablesGlobales.Epocas; epoca++)
			{
				errorEpoca = 0;

				for (int i = 0; i < VariablesGlobales.Entradas.Length; i++)
				{
					propagacion(VariablesGlobales.Entradas[i]);

					calcularError(i);

					retropropagacion(VariablesGlobales.Salidas[i]);
				}

				errorEpoca /= totalPatrones;

				if (verificarErrorNAN(epoca))
					break;

				if (verificarPaciencia(epoca, ref epocasSinMejora))
					break;

				if (verificarErrorMinimo(epoca))
					break;

				epocasAlcanzadas++;
			}

			//guardarDatos();

			MessageBox.Show($"Entrenamiento finalizado.", "Entrenamiento");
		}

		/// <summary>
		/// Realiza la propagación hacia adelante
		/// </summary>
		/// <param name="entradas">Entradas de la red.</param>
		/// <returns>Salida de la red.</returns>
		public double[] propagacion(double[] entradas)
		{
			double[] salidas = new double[capas[capas.Count - 1].neuronas.Count];

			entradas = normalizarDatos(entradas);

			capas[0].calcularActivacion(entradas);

			for (int c = 1; c < capas.Count; c++)
				capas[c].calcularActivacion();

			
			for (int i = 0; i < salidas.Length; i++)
			{
				salidas[i] = capas[capas.Count - 1].neuronas[i].a;
			}

			if (salidas.Length > 1)
			{
				return softmax(salidas);
			}

			return salidas;  
		}

		/// <summary>
		/// Realiza la retropropagación
		/// </summary>
		/// <param name="salidaEsperada">Salida esperada.</param>
		private void retropropagacion(double[] salidaEsperada)
		{
			capas[capas.Count - 1].calcularError(salidaEsperada);

			for (int c = capas.Count - 2; c > 0; c--)
				capas[c].calcularError();

			foreach (var capa in capas)
				capa.actualizarBiasPesos();
		}

		/// <summary>
		/// Normaliza un valor individual
		/// </summary>
		/// <param name="entrada">Valor a normalizar.</param>
		/// <returns>Valor normalizado.</returns>
		public double normalizarDatos(double entrada)
		{
			return (entrada - VariablesGlobales.Min) / (VariablesGlobales.Max - VariablesGlobales.Min);
		}

		/// <summary>
		/// Normaliza un arreglo de valores
		/// </summary>
		/// <param name="entradas">Arreglo de valores a normalizar.</param>
		/// <returns>Arreglo de valores normalizados.</returns>
		public double[] normalizarDatos(double[] entradas)
		{
			double[] resultado = new double[entradas.Length];
			for (int i = 0; i < entradas.Length; i++)
			{
				resultado[i] = normalizarDatos(entradas[i]);
			}
			return resultado;
		}

		/// <summary>
		/// Realiza la predicción de la red.
		/// </summary>
		/// <param name="x">Datos que se van a procesar</param>
		/// <returns>Los datos ya procesados</returns>
		private double[] softmax(double[] x)
		{
			double max = x.Max();
			double[] exponenciales = new double[x.Length];
			double sumaExponenciales = 0;

			for (int i = 0; i < x.Length; i++)
			{
				exponenciales[i] = Math.Exp(x[i] - max);
				sumaExponenciales += exponenciales[i];
			}

			for (int i = 0; i < x.Length; i++)
			{
				exponenciales[i] /= sumaExponenciales;
			}

			return exponenciales;
		}

		/// <summary>
		/// Carga los datos de configuración desde un archivo
		/// </summary>
		/// <returns>True si la carga fue exitosa, False en caso contrario.</returns>
		//public bool cargarDatos()
		//{
		//	try
		//	{
		//		if (!File.Exists(VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos))
		//		{
		//			MessageBox.Show($"El archivo {VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos} no existe.", "Error");
		//			return false;
		//		}

		//		int capaActual = -1;
		//		int neuronaActual = 0;
		//		int pesoActual = 0;

		//		List<string> lineas = archivo.LeerArchivo(VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);

		//		foreach(string linea in lineas)
		//		{
		//			if (linea.StartsWith("Capa"))
		//			{
		//				capaActual++;
		//				neuronaActual = 0;
		//				pesoActual = 0;
		//			}
		//			else if (linea.StartsWith("Peso"))
		//			{
		//				if (capas[capaActual].Tipo != Capa.TipoCapa.Entrada)
		//				{
		//					string[] partes = linea.Split('=');
		//					if (partes.Length != 2)
		//					{
		//						MessageBox.Show($"Formato incorrecto en la línea: {linea}", "Error");
		//						return false;
		//					}

		//					double peso = double.Parse(partes[1].Trim());

		//					if (capaActual >= 0 && capaActual < capas.Length && neuronaActual < capas[capaActual].neuronas.Length)
		//					{
		//						capas[capaActual].neuronas[neuronaActual].Pesos[pesoActual] = peso;
		//						pesoActual++;
		//						if (pesoActual >= capas[capaActual].neuronas[neuronaActual].Pesos.Length)
		//						{
		//							pesoActual = 0;
		//							neuronaActual++;
		//						}
		//					}
		//				}
		//			}
		//			else if (linea.StartsWith("Bias"))
		//			{
		//				if (capas[capaActual].Tipo != Capa.TipoCapa.Entrada)
		//				{
		//					string[] partes = linea.Split('=');
		//					if (partes.Length != 2)
		//					{
		//						MessageBox.Show($"Formato incorrecto en la línea: {linea}", "Error");
		//						return false;
		//					}

		//					double sesgo = double.Parse(partes[1].Trim());

		//					if (capaActual >= 0 && capaActual < capas.Length && neuronaActual < capas[capaActual].neuronas.Length)
		//					{
		//						capas[capaActual].neuronas[neuronaActual].Bias = sesgo;
		//						neuronaActual++;
		//					}
		//				}
		//			}
		//		}

		//		MessageBox.Show("Configuración cargada correctamente.", "Perceptron");
		//		return true;
		//	}
		//	catch (Exception e)
		//	{
		//		MessageBox.Show($"No se ha podido cargar la configuración.\nError:\n {e.Message}\nStackTrace:\n{e.StackTrace}");
		//		return false;
		//	}
		//}

		/// <summary>
		/// Guarda los datos de configuración en un archivo
		/// </summary>
		//public void guardarDatos()
		//{
		//	try
		//	{
		//		for(int c = 0; c < capas.Length; c++)
		//		{
		//			archivo.EscribirArchivo($"Capa {c}:", VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);

		//			for (int j = 0; j < capas[c].neuronas.Length ; j++)
		//			{
		//				for(int p = 0; p < capas[c].neuronas[j].Pesos.Length ; p++)
		//				{
		//					archivo.EscribirArchivo($"Peso = {capas[c].neuronas[j].Pesos[p]}", VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);
		//				}

		//				archivo.EscribirArchivo($"Bias = {capas[c].neuronas[j].Bias}", VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);
		//			}
		//		}

		//		MessageBox.Show("Se ha guardado la configuración correctamente.", "Perceptron");
		//	}
		//	catch (Exception e)
		//	{
		//		MessageBox.Show($"No se ha podido guardar la configuración.\nError:\n {e.Message}");
		//	}
		//}
	}
}