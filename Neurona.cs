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
		public List<Neurona> NeuronasSiguientes;

		/// <summary>
		/// Lista de neuronas de la capa anterior.
		/// </summary>
		public List<Neurona> NeuronasAnteriores;

		/// <summary>
		/// Constructor de la clase Neurona.
		/// </summary>
		/// <param name="nombre">Nombre de la neurona.</param>
		/// <param name="numeroPesos">Número de pesos (conexiones con la capa anterior).</param>
		public Neurona(string name, int numeroNeuronasSiguientes)
		{
			NeuronasSiguientes = new List<Neurona>();
			NeuronasAnteriores = new List<Neurona>();
			Nombre = name;

			if (numeroNeuronasSiguientes > 0)
			{
				Pesos = new double[numeroNeuronasSiguientes];
				Random rand = new Random();
				for (int w = 0; w < numeroNeuronasSiguientes; w++)
				{
					Pesos[w] = rand.NextDouble(); 
				}
				Bias = rand.NextDouble();
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
		/// Método para agregar una neurona e interconectarla con las otras neuronas.
		/// </summary>
		/// <param name="tipo"></param>
		public void agregarNeurona(bool tipo)
		{
			if (tipo)
			{
				NeuronasSiguientes.Add(new Neurona($"Neurona_{NeuronasSiguientes.Count}", NeuronasSiguientes.Count));
			}
			else
			{
				NeuronasAnteriores.Add(new Neurona($"Neurona_{NeuronasAnteriores.Count}", NeuronasAnteriores.Count));
			}
		}
	}
}
