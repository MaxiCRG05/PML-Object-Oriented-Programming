using System;

namespace Perceptron_Multicapa_Colores
{
	/// <summary>
	/// Clase Neuronas para manejar las neuronas del PML de cada capa.
	/// </summary>
	public class Neurona
	{
		/// <summary>
		/// Variable para obtener un número aleatorio.
		/// </summary>
		private static Random rand = new Random();

		/// <summary>
		/// Nombre de la neurona.
		/// </summary>
		public string Nombre { get; set; }

		/// <summary>
		/// Pesos de la neurona, cada peso dependiendo de la capa siguiente
		/// </summary>
		public double[] Pesos { get; set; } 

		/// <summary>
		/// Bias de cada neurona.
		/// </summary>
		public double Bias { get; set; }
		
		/// <summary>
		/// Delta o error de cada neurona.
		/// </summary>
		public double Delta { get; set; } 

		/// <summary>
		/// Salida predicha de cada neurona.
		/// </summary>
		public double Salida { get; set; } 

		/// <summary>
		/// Constructor de la clase Neurona.
		/// </summary>
		/// <param name="nombre">Nombre de la neurona.</param>
		/// <param name="numeroPesos">Número de pesos (conexiones con la capa siguiente).</param>
		public Neurona(string nombre, int numeroPesos)
		{
			
			Nombre = nombre;
			if (numeroPesos > 0)
			{
				Pesos = new double[numeroPesos];

				for (int i = 0; i < numeroPesos; i++)
				{
					Pesos[i] = (double)(rand.NextDouble() - 0.5) * Math.Sqrt(2.0 / numeroPesos);
				}
				Bias = (double)rand.NextDouble() - 0.5f;
			}
			else
			{
				Pesos = null;
				Bias = 0;
			}
			Delta = 0;
			Salida = 0;
		}

		/// <summary>
		/// Método para actualizar los pesos de la neurona durante la retropropagación.
		/// </summary>
		/// <param name="tasaAprendizaje">Tasa de aprendizaje.</param>
		/// <param name="entradas">Entradas de la capa anterior.</param>
		public void ActualizarPesos(double tasaAprendizaje, double[] entradas)
		{
			if (Pesos != null && entradas.Length == Pesos.Length)
			{
				for (int i = 0; i < Pesos.Length; i++)
				{
					Pesos[i] += tasaAprendizaje * Delta * entradas[i];
				}
				Bias += tasaAprendizaje * Delta;
			}
		}
	}
}
