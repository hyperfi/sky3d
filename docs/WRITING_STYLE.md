# Scientific Writing Style Guide

Use this document as a standing instruction whenever drafting, revising, or polishing scientific manuscripts, proceedings, abstracts, captions, referee responses, or technical notes in this repository.

The goal is not merely grammatical correctness. The writing must read like a paper written by an experienced researcher who understands **which result matters and why**.

---

## 1. Core rule: physics first, narrative first

Write around the scientific argument, not around the order in which calculations were performed.

Preferred flow:

**physical question -> result -> interpretation -> essential evidence -> limitation**

Avoid:

**run -> number -> diagnostic -> number -> convergence check -> number -> next run**

A manuscript is not a log file.

Before writing a paragraph, ask:

1. What is the physical statement?
2. Why does it matter?
3. What is the minimum evidence needed to support it?
4. What limitation must be stated to keep the claim precise?

If a number does not materially help answer one of these questions, omit it from the main text.

---

## 2. Tone

The style should be:

- concise;
- professional;
- technically precise;
- physics-led rather than procedure-led;
- confident where supported;
- conservative where limitations exist;
- free of promotional language;
- natural scientific prose rather than AI-generated exposition.

Prefer direct statements.

Good:

> The low-energy monopole structure is accompanied by a strong $K=0$ quadrupole response, identifying it with the deformation-induced mixed mode.

Avoid:

> It is interesting to note that the calculations reveal a very clear and remarkable correspondence between the monopole and quadrupole channels.

Avoid filler such as:

- “It is worth noting that...”
- “Interestingly...”
- “Remarkably...”
- “Importantly...”
- “It can clearly be seen that...”
- “As is evident from...”
- “These findings underscore...”
- “This opens exciting avenues...”
- “Taken together, these results demonstrate the power of...”

Let the physics carry the emphasis.

---

## 3. Result first, numbers second

State the physics first and use only the minimum numbers needed.

Preferred:

> Both EDFs place the mixed $E0-E2$ mode at nearly the same energy and preserve its separation from the higher-energy main ISGMR. The calculated centroids remain within about $0.6$ MeV of experiment.

Avoid long strings of exact values unless those exact values are the scientific point.

Detailed diagnostics belong in tables, captions, appendices, supplementary material, or internal notes whenever possible.

---

## 4. Do not write the paper as a validation report

Numerical validation is necessary, but usually summarize it qualitatively.

Preferred:

> The mixed response is reciprocal within numerical accuracy, remains unchanged under a reduced boost amplitude, and is stable at the level of energy-window observables.

Avoid dumping mismatch percentages, fitted scale factors, convergence shifts, or file-level diagnostics into the main narrative unless the validation itself is a central result.

---

## 5. Explain what the result means physically

Never stop at what was computed.

Weak:

> The $E0$, $E2$, and cross-response peaks occur at the same energy.

Preferred:

> The coincidence of the diagonal $E0$, $K=0$ $E2$, and off-diagonal responses shows that a monopole perturbation directly excites the quadrupole degree of freedom associated with the mixed mode.

Always move from **observable -> interpretation**.

---

## 6. Introduction

Keep the Introduction short and purposeful.

Preferred structure:

1. establish the physical problem;
2. state what the latest experiment/theory already established;
3. identify the specific unresolved or complementary question;
4. state exactly what this work adds.

Do not turn the Introduction into a broad literature review unless required.

Do not exaggerate novelty by understating previous work.

Preferred novelty framing:

> QRPA identifies the mixed modes through simultaneous IS0 and $K=0$ IS2 strength; here we complement this eigenmode picture by calculating the reciprocal off-diagonal response directly in TDHF.

Avoid claims like “for the first time” unless verified carefully.

The reader should understand the paper’s punchline before reaching the Method section.

---

## 7. Method

Explain the physical workflow before numerical details.

Preferred order:

**state -> perturbation -> evolution/solver -> observable -> response quantity**

Example:

> After preparing the deformed HF minimum, weak monopole and quadrupole boosts are applied independently. The TDHF state is then evolved while both multipole moments are monitored, allowing diagonal and off-diagonal responses to be extracted.

Then give only the numerical setup needed for reproducibility or interpretation.

Usually keep:

- model/functional;
- basis/grid if important;
- operator definition;
- propagation time if relevant to resolution;
- response definition;
- smoothing prescription when it affects interpretation.

Usually omit from a short paper unless essential:

- zero-padding details;
- every timestep diagnostic;
- exact file-processing sequence;
- redundant convergence numbers;
- implementation trivia that does not affect the physics.

A Method section should explain **how the physical result is obtained**, not reproduce the README.

---

## 8. Results and Discussion

This section must be interpretive.

Preferred paragraph structure:

**observation -> physical interpretation -> supporting comparison -> limitation**

Example:

> The low-energy monopole structure is accompanied by a strong $K=0$ quadrupole response for both EDFs. The off-diagonal response peaks in the same region, showing that the monopole perturbation directly drives the quadrupole degree of freedom associated with the deformation-induced mixed mode. The reciprocal calculation gives the same response within numerical accuracy, confirming linear cross-channel dynamics.

Use figures to carry numerical detail. Use prose to tell the reader what the figures mean.

Do not narrate the sequence of runs.

---

## 9. Conclusion

Do not repeat the Results section or list validation checks.

Preferred structure:

1. one sentence stating the physical result;
2. one sentence stating its significance or experimental connection;
3. one short limitation if needed.

Example:

> TDHF time evolution shows that an $E0$ excitation of deformed $^{24}$Mg directly induces the $K=0$ quadrupole motion associated with the monopole--quadrupole mixed mode. The effect is robust across the EDFs considered and is consistent with the experimentally identified low-energy MQC region. The detailed fragmentation remains beyond mean-field TDHF.

Avoid checklist conclusions such as:

> Reciprocity, linearity, two-EDF persistence, box convergence, half-amplitude scaling, and stable window observables establish...

Those are supporting diagnostics, not the conclusion.

---

## 10. Abstract

The abstract should contain:

1. the problem;
2. what was done;
3. the central result;
4. why it matters.

Do not catalogue computational parameters.

Prefer one strong quantitative statement rather than many minor numbers.

The reader should understand the contribution after one reading.

---

## 11. Figures and captions

Captions should state:

- what is plotted;
- what the panels represent;
- essential normalization/smoothing information;
- any distinction needed to interpret the figure correctly.

Do not turn captions into a second Results section.

If different curves use different operator normalizations, state it.

If widths are artificially smoothed, state it.

If experimental data were shifted, scaled, normalized, or digitized, state it explicitly.

Never hide transformations applied to theory or data.

---

## 12. Numerical precision

Do not report more digits than the physics supports.

Use exact values when:

- required for reproducibility;
- needed to demonstrate a numerical property;
- the difference between close values is physically relevant.

Otherwise summarize.

Prefer:

> within about $0.6$ MeV of experiment

instead of four separate centroid differences to three decimal places.

A manuscript should not look more precise than the underlying model.

---

## 13. Limitations

State limitations directly and without apologetic language.

Preferred:

> TDHF reproduces the gross two-region structure but not the observed fine fragmentation, indicating missing correlations beyond mean field.

Avoid:

> Unfortunately, due to limitations of our method, we are unable to reproduce...

A limitation defines the domain of the claim; it does not invalidate the paper.

Distinguish carefully between:

- robust physics;
- model dependence;
- numerical non-convergence;
- physics outside the model.

---

## 14. Novelty claims

Never inflate novelty.

Before using words such as:

- first;
- new;
- demonstrates;
- establishes;
- direct evidence;

verify that the literature supports them.

Prefer technically exact novelty statements:

- “we calculate the off-diagonal response directly”;
- “we complement the eigenmode picture with time-domain dynamics”;
- “we test the same mechanism with a different many-body framework”;
- “we expose the cross-channel susceptibility rather than separate diagonal strengths.”

The difference between prior work and the present work should be exact, not rhetorical.

---

## 15. Comparison with experiment

Do not write only “good agreement.”

State what agrees and what does not.

Preferred:

> Both EDFs recover the separation between the low-energy MQC and higher-energy ISGMR regions, while overconcentrating the low-energy strength and missing the measured fragmentation.

This is better than:

> The theoretical results agree reasonably well with experiment.

Never fit, scale, shift, or renormalize theory to experiment without stating it explicitly.

---

## 16. Equations and notation

Use equations when they clarify the physics or define a quantity central to the argument.

Do not include equations simply because they are available.

Define all nonstandard notation.

For response functions distinguish clearly between:

- susceptibility;
- strength function;
- signed mixed response;
- absolute magnitude;
- real and imaginary parts.

Do not call a signed or complex cross response a positive-definite “strength” unless the definition justifies it.

---

## 17. LaTeX conventions

Keep source readable and conventional.

Prefer:

```latex
$^{24}$Mg
Fig.~\ref{fig:key}
Ref.~\cite{key}
10--18 MeV
```

Use math mode for mathematical symbols and minus signs.

Avoid excessive parentheses and unnecessary custom macros.

Do not create macros for terms that appear only once or twice.

---

## 18. Proceedings papers and strict page limits

Prioritize, in this order:

1. scientific question;
2. central result;
3. one convincing figure;
4. physical interpretation;
5. essential method;
6. essential validation;
7. limitation.

Cut first:

- long background;
- exhaustive diagnostics;
- exact convergence numbers;
- values already visible in figures;
- implementation details;
- secondary tests.

Do not sacrifice the scientific narrative to preserve every technical detail.

A short proceedings paper should read like a **compact physics argument**, not a compressed full paper.

---

## 19. Full journal papers

More space should be used for:

- deeper physical interpretation;
- systematic comparisons;
- uncertainty and model-dependence analysis;
- derivations;
- additional observables;
- stronger connection to prior literature.

Do not use extra space merely to dump more numerical output.

---

## 20. Internal notes versus manuscript prose

Internal notes may contain:

- file names;
- commands;
- run IDs;
- exact convergence tables;
- diagnostics;
- failed tests;
- numerical checks.

The manuscript should contain only what advances or supports the scientific argument.

When drafting from `STATUS.md`, logs, notebooks, or scripts, **translate evidence into scientific prose**. Do not copy their reporting style into the paper.

---

## 21. Voice

Use active scientific prose where natural:

> We calculate...
> We find...
> We compare...

Passive voice is also fine where the actor is irrelevant:

> The response is obtained by Fourier transformation.

Do not force either active or passive voice uniformly.

Avoid repeatedly beginning consecutive sentences with “We.”

---

## 22. Editing rule: cut before adding

When revising under a length constraint:

1. identify the scientific point of each paragraph;
2. remove numbers already conveyed by figures;
3. collapse validation details into qualitative statements;
4. remove repeated interpretation;
5. shorten background already covered by citations;
6. only then add missing physics discussion.

Do not solve page-limit problems by shrinking fonts or overcrowding figures before tightening the prose.

---

## 23. Transformation examples

### Log-style

> The SLy5 calculation gives a peak at 16.857 MeV, while SkM* gives 16.590--16.598 MeV. The forward and reverse responses have relative $L^2$ mismatches of 0.108\% and 0.097\%. The scale factors are 1.00016 and 0.99996. Halving the boost changes the response by less than 0.01\%.

### Preferred

> For both EDFs, the diagonal and off-diagonal responses identify the same low-energy mixed mode. The independently calculated forward and reverse responses satisfy reciprocity within numerical accuracy and remain unchanged under a reduced boost, confirming a linear cross-channel response.

### Log-style

> The $32^3$ box shifts the centroid by 0.015 MeV, the integral by 0.15\%, and the mixed-response integral by 0.01\%, whereas the pointwise spectrum differs by 79\%.

### Preferred

> Energy-window centroids and integrated responses are stable against the box-size test, whereas individual narrow lines are not. We therefore base the physical interpretation on collective window observables rather than fine spectral structure.

### Generic conclusion

> The calculations were performed using two EDFs and several numerical tests. The results show good agreement with experiment and demonstrate the usefulness of the method.

### Preferred

> The calculation identifies the low-energy structure as a dynamically coupled mode and reproduces its separation from the higher-energy resonance. Its detailed fragmentation, however, remains beyond the present mean-field description.

---

## 24. Final self-check

Before accepting manuscript text, verify:

- Does the first sentence contain a scientific point?
- Is the paragraph discussing physics rather than reporting workflow?
- Are all numbers necessary?
- Does every numerical detail support a claim?
- Is previous literature represented fairly?
- Is novelty stated exactly rather than rhetorically?
- Are limitations explicit?
- Is any sentence generic enough to appear in an unrelated paper? If yes, rewrite it.
- Can a sentence be removed without losing information? If yes, remove it.
- Does the prose sound like a researcher interpreting results rather than an automated system summarizing output?

When in doubt, prefer **physics over bookkeeping, interpretation over enumeration, and precision over decoration**.
