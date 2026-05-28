Table 10 confirms, the wall-clock advantage of DPSGD does not arise from
cheaper iterations but from a substantially smaller number of iterations to reach
a given target lossΓÇöa direct reflection of the improved conditioning produced by
damping and temporal smoothing. This trade-off embodies the canonical second-order
argument: greater computational effort per step, less aggregate effort overall.
Across all four benchmarks the qualitative pattern is consistent. Damping stabilizes
the early training dynamics, during which the curvature estimate is most stochastically
corrupted; temporal smoothing further reduces the variance of the factor estimates;
and the LevenbergΓÇôMarquardt regulated╧äschedule shrinks the regularizer as the local
quadratic model becomes reliable. The combined effect is a curvature-aware step that
retains the conditioning benefits of PSGD while shedding its early-phase fragility.
5 Limitations and Future Work
The contribution of this study is mathematical and diagnostic, not architectural;
several limitations follow directly from that scope.
22
Scope and architectural generality.
All empirical results were obtained on small fully connected, LeNet-style convo-
lutional, and small recurrent networks on MNIST, FashionMNIST, the CURVES
autoencoder, and the synthetic addition task, on a single NVIDIA T4 GPU. These
benchmarks are deliberately chosen as standard reference workloads for the verifi-
cation of second-order optimizers, isolating the mathematical behaviour of damped
Kronecker preconditioning from confounding variables such as distributed training,
mixed-precision arithmetic, and optimizer-state sharding. They are not surrogates for
state-of-the-art training, and no claim is made that the reported wall-clock reductions
transfer without further analysis to Transformer-scale models, large vision backbones,
or contemporary multimodal pipelines. All experiments use the dense Kronecker-factor
form of the preconditioner; PSGD admits diagonal, reverse-arrow-head, and other
sparsity patterns [32] that substantially alter memory and compute behaviour at scale,
and the interaction of damping with such sparse factorizations has not been character-
ized empirically here, although the underlying derivations of Section 3 extend formally
without modification. The principal directions for extending these scope limits are
scaling to ResNets [41], DenseNets [42], and Vision Transformers, in which sparser
Kronecker variants will likely be required, and a distributed multi-GPU implemen-
tation in the manner of distributed Shampoo [29], in which the per-layer trace ratio
in (41) and the LM update (33) both reduce to scalar all-reduce operations.
Empirical extensions.
The DPSGD schedule depends on the preconditioner update periodT 1, damping
update periodT 2, initial damping╧ä 0, trust-region thresholds 3/4 and 1/4, and aver-
aging weight╬▓. The first four are inherited from PSGD [3] and K-FAC [4] with
well-documented robustness; only╬▓varies across our benchmarks (╬▓= 0.9 on
MNIST,╬▓= 0.7 elsewhere), selected on the basis of the variance-reduction anal-
ysis of Section 3.3. Three focused empirical extensions, each a single-benchmark
sweep of comparable scope, would strengthen these results: a component-wise ablation
isolating Tikhonov damping from weighted averaging in the four-cell configuration
{PSGD, PSGD+damping, PSGD+averaging, DPSGD}; a sensitivity sweep over╬▓Γêê
{0.5,0.6,0.7,0.8,0.9,1.0}to empirically validate the variance-reduction analysis; and
a multi-seed replication (e.g., five seeds per configuration) reporting medians and
inter-quartile ranges to quantify across-seed variance of the wall-clock-to-target-loss
measurements.
Theoretical and comparative gaps.
The present work develops DPSGD from second-order Taylor and trust-region prin-
ciples but does not derive non-convex convergence rates. A rigorous bound on the
expected stationarity gap under the damped, temporally smoothed update remains
future work; of particular int
---
(29) :  t automatically, we employ the LevenbergΓÇô
Marquardt trust-region ratio [33, 36, 37], which compares the model-predicted
reduction in the loss to the actually observed reduction:
╧ü= J(╬╕+╬┤)ΓêÆJ(╬╕)
M(╬┤) ,(29)
where the model-predicted decrease in the loss for layerlis computed from the damped
quadratic model (25) as
Ml(╬┤l) =ΓêÆΓêç lJ(╬╕) Γèñ ╬┤l ΓêÆ 1
2 ╬┤Γèñ
l (Pl +╧ä l I) ΓêÆ1 ╬┤l.(30)
This is the canonical predicted-reduction form used in trust-region methods [33]:
a positive linear term inΓêÆΓêçJ Γèñ╬┤(representing the gradient-aligned descent) minus
a non-negative quadratic correction. Substituting the proposed step╬┤ l =ΓêÆ(P l +
╧äl I)Γêç lJ(╬╕) from (24) yields the closed form
Ml(╬┤l) =Γêç lJ(╬╕) Γèñ (Pl +╧ä l I)Γêç lJ(╬╕)ΓêÆ 1
2 ΓêçlJ(╬╕) Γèñ (Pl +╧ä l I)Γêç lJ(╬╕)
= 1
2 ΓêçlJ(╬╕) Γèñ (Pl +╧ä l I)Γêç lJ(╬╕).(31)
Ml is non-negative wheneverP l +╧ä l Iis symmetric positive definite, which is
guaranteed by construction sinceP l Γ¬░0 and╧ä l ΓëÑ0.
For multilayer networks, the layer-wise predicted decreases{Ml}L
l=1 are aggregated
into a single scalar
---
(36) : zation delivers a multiplication-count reduction of three
to six orders of magnitude per layer.
The damping operations added by DPSGD do not change this asymptotic com-
plexity. The first formulation (36) adds a single scalarΓÇômatrix multiplication of cost
O(mn) per layer; the second formulation (46) adds two scalar-shifted matrix multi-
plications, each of costO(m 2n+mn 2) =O(mn(m+n)), which dominates only by
a constant factor. The weighted-averaging update (47) costsO(m 2 +n 2) per layer.
Thus the total per-iteration cost of DPSGD isO(mn(m+n)) multiplications and
O(m2 +n 2) storage per layer, identical to undamped PSGD in asymptotic terms.
Throughout this paper, the vectorization operator vec(┬╖) stacks the rows of its
argument in the order consistent with PyTorchΓÇÖs tensor storage [34], so that the mixed-
product identity takes the form (P R ΓèùP L) vec(G) = vec(P L G P Γèñ
R ). The Kronecker
factor on the right of the product acts on the column dimension of the gradient.
For convolutional
---
(46) : The damping operations added by DPSGD do not change this asymptotic com-
plexity. The first formulation (36) adds a single scalarΓÇômatrix multiplication of cost
O(mn) per layer; the second formulation (46) adds two scalar-shifted matrix multi-
plications, each of costO(m 2n+mn 2) =O(mn(m+n)), which dominates only by
a constant factor. The weighted-averaging update (47) costsO(m 2 +n 2) per layer.
Thus the total per-iteration cost of DPSGD isO(mn(m+n)) multiplications and
O(m2 +n 2) storage per layer, identical to undamped PSGD in asymptotic terms.
Throughout this paper, the vectorization operator vec(┬╖) stacks the rows of its
argument in the order consistent with PyTorchΓÇÖs tensor storage [34], so that the mixed-
product identity takes the form (P R ΓèùP L) vec(G) = vec(P L G P Γèñ
R ). The Kronecker
factor on the right of the product acts on the column dimension of the gradient.
For convolutional layers, in which the weight tensor is four-dimensional, the reshape
(fout, f in, k 1, k 2)ΓêÆ ΓåÆ(f
---
(47) : per layer; the second formulation (46) adds two scalar-shifted matrix multi-
plications, each of costO(m 2n+mn 2) =O(mn(m+n)), which dominates only by
a constant factor. The weighted-averaging update (47) costsO(m 2 +n 2) per layer.
Thus the total per-iteration cost of DPSGD isO(mn(m+n)) multiplications and
O(m2 +n 2) storage per layer, identical to undamped PSGD in asymptotic terms.
Throughout this paper, the vectorization operator vec(┬╖) stacks the rows of its
argument in the order consistent with PyTorchΓÇÖs tensor storage [34], so that the mixed-
product identity takes the form (P R ΓèùP L) vec(G) = vec(P L G P Γèñ
R ). The Kronecker
factor on the right of the product acts on the column dimension of the gradient.
For convolutional layers, in which the weight tensor is four-dimensional, the reshape
(fout, f in, k 1, k 2)ΓêÆ ΓåÆ(f out, f in ┬╖k 1 ┬╖k 2)
reduces the tensor to two dimensions before applying (Eq. 17, . An alternative con-
struction expresses the preconditioner as a Kronecker produc
---
(48) : ion
rule, and exponential temporal smoothing of the factorsΓÇöyields the DPSGD algorithm
summarized in Algorithm 2. A step-size clipping rule of the form
╬▒adj = min
 
╬╜pP ΓêÑΓêçJ(╬╕)ΓêÑ 2 ,1
!
├ù╬▒, ╬╜= 0.1
ΓêÜ
N ,(48)
is retained from the PSGD reference implementation [3] to handle pathological gradi-
ent norms during early training, whereNis the total number of trainable parameters.
The
ΓêÜ
Nscaling keeps the per-coordinate gradient budget approximately constant
across network sizes; the default coefficient 0.1 performed reliably across all four
benchmarks without further tuning. Clipping acts orthogonally to damping: damping
regulates the spectrum of the curvature surrogate, whereas clipping bounds the raw
gradient signal before it is preconditioned.
The Kronecker factor update invoked in Algorithm 2 derives from the PSGD crite-
rion (9). SettingA=Q ╬┤gandB=Q ΓêÆΓèñ ╬┤╬╕, the gradient offadmits the symmetric
form
Γêçf= triu
 
A AΓèñ ΓêÆB B Γèñ
.(49)
Under the Kronecker factorizationQ=Q L ΓèùQ R, the same operation
---
Algorithm 2 : torization of the pre-
conditioner, Tikhonov damping regulated by the LevenbergΓÇôMarquardt trust-region
rule, and exponential temporal smoothing of the factorsΓÇöyields the DPSGD algorithm
summarized in Algorithm 2. A step-size clipping rule of the form
╬▒adj = min
 
╬╜pP ΓêÑΓêçJ(╬╕)ΓêÑ 2 ,1
!
├ù╬▒, ╬╜= 0.1
ΓêÜ
N ,(48)
is retained from the PSGD reference implementation [3] to handle pathological gradi-
ent norms during early training, whereNis the total number of trainable parameters.
The
ΓêÜ
Nscaling keeps the per-coordinate gradient budget approximately constant
across network sizes; the default coefficient 0.1 performed reliably across all four
benchmarks without further tuning. Clipping acts orthogonally to damping: damping
regulates the spectrum of the curvature surrogate, whereas clipping bounds the raw
gradient signal before it is preconditioned.
The Kronecker factor update invoked in Algorithm 2 derives from the PSGD crite-
rion (9). SettingA=Q ╬┤gandB=Q ΓêÆΓèñ ╬┤╬╕, the gradient offadmits the symmetric
---
