namespace Perceptron_Multicapa_Colores
{
    partial class Form1
    {
        /// <summary>
        /// Variable del diseñador necesaria.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Limpiar los recursos que se estén usando.
        /// </summary>
        /// <param name="disposing">true si los recursos administrados se deben desechar; false en caso contrario.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Código generado por el Diseñador de Windows Forms

        /// <summary>
        /// Método necesario para admitir el Diseñador. No se puede modificar
        /// el contenido de este método con el editor de código.
        /// </summary>
        private void InitializeComponent()
        {
            this.btnProbar = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.respuestaNeurona = new System.Windows.Forms.Label();
            this.button2 = new System.Windows.Forms.Button();
            this.resultado = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.registroColores = new System.Windows.Forms.RichTextBox();
            this.btnLimpiar = new System.Windows.Forms.Button();
            this.btnGuardar = new System.Windows.Forms.Button();
            this.btnLimpiarPesos = new System.Windows.Forms.Button();
            this.btnImagen1 = new System.Windows.Forms.Button();
            this.btnImagen2 = new System.Windows.Forms.Button();
            this.btnImagenPersonalizada = new System.Windows.Forms.Button();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.SuspendLayout();
            // 
            // btnProbar
            // 
            this.btnProbar.Enabled = false;
            this.btnProbar.Location = new System.Drawing.Point(12, 407);
            this.btnProbar.Name = "btnProbar";
            this.btnProbar.Size = new System.Drawing.Size(776, 30);
            this.btnProbar.TabIndex = 2;
            this.btnProbar.Text = "PROBAR";
            this.btnProbar.UseVisualStyleBackColor = true;
            this.btnProbar.Click += new System.EventHandler(this.btnProbar_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.ForeColor = System.Drawing.Color.Black;
            this.label2.Location = new System.Drawing.Point(180, 9);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(138, 25);
            this.label2.TabIndex = 4;
            this.label2.Text = "                     ";
            this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Microsoft Sans Serif", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.Location = new System.Drawing.Point(12, 9);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(162, 25);
            this.label3.TabIndex = 6;
            this.label3.Text = "Color estimado:";
            // 
            // respuestaNeurona
            // 
            this.respuestaNeurona.AutoSize = true;
            this.respuestaNeurona.Font = new System.Drawing.Font("Microsoft Sans Serif", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.respuestaNeurona.Location = new System.Drawing.Point(182, 97);
            this.respuestaNeurona.Name = "respuestaNeurona";
            this.respuestaNeurona.Size = new System.Drawing.Size(0, 25);
            this.respuestaNeurona.TabIndex = 7;
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(12, 371);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(776, 30);
            this.button2.TabIndex = 8;
            this.button2.Text = "ENTRENAR";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.MouseClick += new System.Windows.Forms.MouseEventHandler(this.button2_MouseClick);
            // 
            // resultado
            // 
            this.resultado.AutoSize = true;
            this.resultado.Font = new System.Drawing.Font("Microsoft Sans Serif", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.resultado.Location = new System.Drawing.Point(188, 97);
            this.resultado.Name = "resultado";
            this.resultado.Size = new System.Drawing.Size(30, 25);
            this.resultado.TabIndex = 9;
            this.resultado.Text = "   ";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Location = new System.Drawing.Point(14, 60);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(204, 25);
            this.label1.TabIndex = 12;
            this.label1.Text = "Registro de colores:";
            // 
            // registroColores
            // 
            this.registroColores.Location = new System.Drawing.Point(17, 97);
            this.registroColores.Name = "registroColores";
            this.registroColores.Size = new System.Drawing.Size(199, 208);
            this.registroColores.TabIndex = 13;
            this.registroColores.Text = "";
            // 
            // btnLimpiar
            // 
            this.btnLimpiar.Location = new System.Drawing.Point(223, 281);
            this.btnLimpiar.Name = "btnLimpiar";
            this.btnLimpiar.Size = new System.Drawing.Size(75, 23);
            this.btnLimpiar.TabIndex = 14;
            this.btnLimpiar.Text = "LIMPIAR";
            this.btnLimpiar.UseVisualStyleBackColor = true;
            this.btnLimpiar.MouseClick += new System.Windows.Forms.MouseEventHandler(this.button1_MouseClick);
            // 
            // btnGuardar
            // 
            this.btnGuardar.Location = new System.Drawing.Point(378, 267);
            this.btnGuardar.Name = "btnGuardar";
            this.btnGuardar.Size = new System.Drawing.Size(116, 38);
            this.btnGuardar.TabIndex = 15;
            this.btnGuardar.Text = "GUARDAR CONFIGURACIÓN";
            this.btnGuardar.UseVisualStyleBackColor = true;
            this.btnGuardar.MouseClick += new System.Windows.Forms.MouseEventHandler(this.btnGuardar_MouseClick);
            // 
            // btnLimpiarPesos
            // 
            this.btnLimpiarPesos.Location = new System.Drawing.Point(378, 238);
            this.btnLimpiarPesos.Name = "btnLimpiarPesos";
            this.btnLimpiarPesos.Size = new System.Drawing.Size(116, 23);
            this.btnLimpiarPesos.TabIndex = 16;
            this.btnLimpiarPesos.Text = "LIMPIAR PESOS";
            this.btnLimpiarPesos.UseVisualStyleBackColor = true;
            this.btnLimpiarPesos.MouseClick += new System.Windows.Forms.MouseEventHandler(this.btnLimpiarPesos_MouseClick);
            // 
            // btnImagen1
            // 
            this.btnImagen1.Location = new System.Drawing.Point(378, 30);
            this.btnImagen1.Name = "btnImagen1";
            this.btnImagen1.Size = new System.Drawing.Size(116, 23);
            this.btnImagen1.TabIndex = 17;
            this.btnImagen1.Text = "IMAGEN 1";
            this.btnImagen1.UseVisualStyleBackColor = true;
            this.btnImagen1.MouseClick += new System.Windows.Forms.MouseEventHandler(this.btnImagen1_MouseClick);
            // 
            // btnImagen2
            // 
            this.btnImagen2.Location = new System.Drawing.Point(378, 61);
            this.btnImagen2.Name = "btnImagen2";
            this.btnImagen2.Size = new System.Drawing.Size(116, 23);
            this.btnImagen2.TabIndex = 18;
            this.btnImagen2.Text = "IMAGEN 2";
            this.btnImagen2.UseVisualStyleBackColor = true;
            this.btnImagen2.MouseClick += new System.Windows.Forms.MouseEventHandler(this.btnImagen2_MouseClick_1);
            // 
            // btnImagenPersonalizada
            // 
            this.btnImagenPersonalizada.Location = new System.Drawing.Point(378, 97);
            this.btnImagenPersonalizada.Name = "btnImagenPersonalizada";
            this.btnImagenPersonalizada.Size = new System.Drawing.Size(116, 40);
            this.btnImagenPersonalizada.TabIndex = 19;
            this.btnImagenPersonalizada.Text = "IMAGEN PERSONALIZADA";
            this.btnImagenPersonalizada.UseVisualStyleBackColor = true;
            this.btnImagenPersonalizada.MouseClick += new System.Windows.Forms.MouseEventHandler(this.btnImagenPersonalizada_MouseClick);
            // 
            // pictureBox1
            // 
            this.pictureBox1.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.pictureBox1.Cursor = System.Windows.Forms.Cursors.Cross;
            this.pictureBox1.Location = new System.Drawing.Point(513, 30);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(275, 275);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox1.TabIndex = 11;
            this.pictureBox1.TabStop = false;
            this.pictureBox1.MouseClick += new System.Windows.Forms.MouseEventHandler(this.pictureBox1_MouseClick);
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 449);
            this.Controls.Add(this.btnImagenPersonalizada);
            this.Controls.Add(this.btnImagen2);
            this.Controls.Add(this.btnImagen1);
            this.Controls.Add(this.btnLimpiarPesos);
            this.Controls.Add(this.btnGuardar);
            this.Controls.Add(this.btnLimpiar);
            this.Controls.Add(this.registroColores);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.pictureBox1);
            this.Controls.Add(this.resultado);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.respuestaNeurona);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.btnProbar);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private System.Windows.Forms.Button btnProbar;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label respuestaNeurona;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.Label resultado;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.RichTextBox registroColores;
        private System.Windows.Forms.Button btnLimpiar;
        private System.Windows.Forms.Button btnGuardar;
        private System.Windows.Forms.Button btnLimpiarPesos;
        private System.Windows.Forms.Button btnImagen1;
        private System.Windows.Forms.Button btnImagen2;
        private System.Windows.Forms.Button btnImagenPersonalizada;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
    }
}

