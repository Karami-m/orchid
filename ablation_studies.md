# Ablation studies on Conditioning Network Architecture
We have conducted further ablation studies on the conditioning network architecture on associative recall task with vocabulary size of 20 and sequence length of 128:

   - Comparison of Local Conv1D Choices: We evaluated different short depthwise linear convolution architectures in the conditioning network, as outlined in Equation (2). This ablation study compared: I) applying $Conv1D()$ in the spatial domain followed by $Conv1D()$ in the frequency domain (as proposed in Equation (2)), II) applying two layers of $Conv1D()$ in the spatial domain only, and III) applying two layers of $Conv1D()$ in the spectral domain only. The results demonstrated that the proposed method of operating in both spatial and frequency domains (as in Equation (2)) which mixes information from neighboring spatial and spectral tokens shows the best performance.

    Fig A1: Comparison of Local Conv1D Choices: Evaluation of different local convolution options used in the conditioning network. Architectures are:  eqn (2) (1 layer Conv1d in Time + 1 layer in Freq), 2 layer Conv1d in Time, 2 layer Conv1d in Freq, and 3 layer Conv1d in Time + 3 layer in Freq
![W B Chart 2024-08-07, 8_14_27 AM-4](https://github.com/user-attachments/assets/77f92eb7-8b79-4a1a-ab35-cc039d698229)

   - Impact of Different Nonlinear $\sigma()$ Functions: We evaluated various nonlinear $\sigma()$ functions used in the Type II (cross-correlation) conditioning network (Equation (3)). The nonlinearities tested included $\[ Tanh(),~ Sigmoid(),~ Softsign(),~ Softshrink(),~  Identity()\]$, all acting on the magnitude of the argument. The results indicated that dropping this nonlinearity, $\sigma()$ in Equation (3), provides the best performance which is slightly better than $nn.Softshrink()$ and $nn.Tanh()$. Also, among the nonlinearities those that cross zero perform better.  
Moreover, we also observe that Type II conditioning networks with $Identity()$ and $Softshrink()$ have faster convergence than Type I. 

    Fig A2: Comparison of differenet $\sigma()$ on conditioning network of Type II (cross-correlation eqn (3) ) with DCT
![Type II DCT](https://github.com/user-attachments/assets/ef7799f6-fc5b-43c0-a597-cfe870044930)

    Fig A3: Comparison of differenet $\sigma()$ on conditioning network of Type II (cross-correlation eqn (3) ) with DFT
![Type II DFT](https://github.com/user-attachments/assets/ca04c7ce-5fe8-49a7-a5f7-5e9986ebaeba)
