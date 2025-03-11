using System;
using System.Collections.Generic;

namespace Perceptron_Multicapa_Colores
{
	/// <summary>
	/// Clase Neuronas para manejar las neuronas del PML de cada capa.
	/// </summary>
	public class Neurona
	{
		/// <summary>
		/// Nombre de la neurona.
		/// </summary>
		public string Nombre { get; set; }

		/// <summary>
		/// Pesos de la neurona dependiendo de las entradas que reciba.
		/// </summary>
		public double[] Pesos { get; set; }

		/// <summary>
		/// Bias de la neurona.
		/// </summary>
		public double Bias { get; set; }

		/// <summary>
		/// Delta (error) de la neurona.
		/// </summary>
		public double Delta { get; set; }

		/// <summary>
		/// Salida (predicción) de la neurona.
		/// </summary>
		public double Salida { get; set; }

		/// <summary>
		/// Lista de neuronas de la capa siguiente.
		/// </summary>
		public List<Neurona> NeuronasSiguiente;

		/// <summary>
		/// Constructor de la clase Neurona.
		/// </summary>
		/// <param name="nombre">Nombre de la neurona.</param>
		/// <param name="numeroPesos">Número de pesos (conexiones con la capa anterior).</param>
		public Neurona(string nombre, int numeroNeuronasSiguientes)
		{
			Nombre = nombre;

			if (numeroNeuronasSiguientes > 0)
			{
				Pesos = new double[numeroNeuronasSiguientes];
				Random rand = new Random();
				for (int w = 0; w < numeroNeuronasSiguientes; w++)
				{
					Pesos[w] = rand.NextDouble(); 
				}
				Bias = rand.NextDouble();
				AgregarNeurona(numeroNeuronasSiguientes);
			}
			else
			{
				Pesos = null;
				Bias = 0;
			}
			Delta = 0;
			Salida = 0;
		}

		private void AgregarNeurona(int numeroNeuronasSiguientes)
		{
			NeuronasSiguiente = new List<Neurona>();
			for (int i = 0; i < numeroNeuronasSiguientes; i++)
			{
				NeuronasSiguiente.Add(new Neurona("Neurona " + i, 0));
			}
		}
	}
}
