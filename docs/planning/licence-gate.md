# Licence Gate

Read from the primary texts on 2026-11-03, before any code, because a "no"
here could stop the project. **Not legal advice.** Items marked *lawyer* or
*vendor* need one before any commercial use. No cracked or trial copies are
used for any of this.

## Summary

| Engine | Licence | Train a model on its outputs and **publish** it (portfolio) | **Sell** that model | Run it inside a paid service | Answer |
|---|---|---|---|---|---|
| **OpenSees** | UC Regents copyright, custom | Yes: "internal purposes" use by an individual, not redistributed | *Probably*, the licence is silent on outputs; *vendor* question | **No** without a UC licence | Settled for the portfolio |
| **OpenSeesPy** | Same, plus its own note | Yes, as above | As above | **No**: its note names cloud services | Settled for the portfolio |
| **OpenQuake engine** | AGPL-3.0 | Yes | Yes for outputs; the code's AGPL terms bind only code | Only if our service's source is released under the AGPL | Settled; kept out of v1 |
| **GEM hazard models and data** | Per product: CC BY-SA or CC BY-NC-SA | Only the CC BY-SA ones, with share-alike | NC products: **no** without a GEM licence | Same | Check each product |
| **ETABS** | CSI EULA (updated 2026-08-18) | We can't make ETABS outputs ourselves: we hold no licence, and trial and academic licences ban this use | Needs our own commercial licence; outputs not addressed; *vendor* | **No**: the EULA bars third-party and pooled access | Add-on on the customer's own licence looks allowed; *vendor* |

## OpenSees

**Text** ([COPYRIGHT](https://github.com/OpenSees/OpenSees/blob/master/COPYRIGHT)):
the Regents grant use "without fee and without a written license agreement"
to (a) educational, research and non-profit entities, for non-commercial
purposes, and (b) other entities, for "internal purposes only". Putting it
into "products for commercial distribution" needs a licence from UC
Berkeley's Office of Technology Licensing.

**OpenSeesPy** ships the same COPYRIGHT file, plus a
[LICENSE.md](https://github.com/zhuminjie/OpenSeesPy/blob/openseespy/pip/openseespy/LICENSE.md)
saying it is free for research, education and internal use. Commercial
redistribution, explicitly including an application or cloud-based service
that uses `import openseespy`, needs a licence (contact: Dr. Minjie Zhu,
Oregon State University).

**What's settled:**
- Arthur running OpenSeesPy on his own machine to make training data is
  internal use. The repo **depends** on `openseespy` and never vendors or
  modifies it, so it redistributes nothing. Our own code can carry
  Arthur's usual MIT licence. The README says that OpenSeesPy has its own
  terms.
- **No runtime path may `import openseespy`.** The screening service loads
  only the trained surrogate. That keeps the service outside OpenSeesPy's
  stated cloud-service case, even for the portfolio demo.

**What isn't:**
- The licence says nothing about **outputs**. Whether a company may sell a
  model trained on outputs it generated "for internal purposes" is a
  question for UC Berkeley's licensing office (IPIRA/OTL) and Dr. Zhu.
  Ask before any sale.
- "Internal purposes" isn't defined. A lawyer should read it for the
  commercial case.

## OpenQuake

**Text:** the engine is
[AGPL-3.0](https://github.com/gem/oq-engine/blob/master/LICENSE). GEM's
[licensing page](https://www.globalquakemodel.org/licensing) says the AGPL
allows any use, including commercial, but AGPL code may not be linked into
closed-source systems, and modified versions must be released.

**Data is separate:** GEM's models and datasets are each under CC BY-SA
(commercial use allowed, share-alike) or CC BY-NC-SA (no commercial use).
The global seismic hazard map raster is NC.

**What's settled:**
- Running the engine as a separate program to compute inputs is fine. Its
  outputs aren't AGPL code.
- **Importing `openquake.hazardlib` into our code makes our code AGPL-bound
  when shared.** That works for an open repo but would also mix with
  OpenSees's non-commercial terms in one program. **v1 doesn't use
  OpenQuake** (ADR-006). Shaking intensity is an input.
- Any GEM dataset used later must be checked product by product. NC
  products stay out of anything sold.

## ETABS (Computers and Structures, Inc.)

**Text:** CSI's
[Software License Agreement](https://www.csiamerica.com/legal/software-license-agreement),
last updated 2026-08-18, under California law:
- §2: one non-transferable, non-sublicensable licence, for use by the
  purchaser or its employees, on one computer at a time.
- §2: **trial or beta copies "may not be used for commercial or professional
  use"**; academic licences are "strictly for non-commercial teaching,
  lesson planning, and research".
- §4(a): no reverse engineering. §4(b): no renting, selling, sublicensing,
  networking or transferring access to a third party. §4(d): no pooling or
  multiplexing, and no letting others use the software indirectly.
- §7: confidential information covers technical information related to
  the software. Whether that reaches **results** isn't stated.
- §8: the software collects machine data.
- The EULA has **no clause on APIs, outputs, benchmarking or machine
  learning**.

**CSI's developer pages** ([developer](https://www.csiamerica.com/developer),
[third-party apps](https://www.csiamerica.com/developer/third-party-applications)):
the API (OAPI) comes with licensed ETABS copies at no extra cost. CSI lists
outside vendors' ETABS add-ons. Support for building integrations is the
paid CSI Developer Network. New releases use cloud sign-in licences only.

**What's settled:**
- **An add-on that runs on the customer's machine, under the customer's own
  licence, reading the customer's own model, matches what CSI lists.** The
  customer is the licensee doing the using. The add-on would read geometry
  and properties and send those (the customer's data, not the software) to
  the screen.
- **We may never host ETABS for customers.** Running it on our server for
  others is the transfer and indirect access that §4(b) and §4(d) bar.
- **Arthur can't make ETABS results himself.** He has no licence, the trial
  bans professional use, and academic copies are for teaching and research
  only. Your friend's student copy is academic, so it can't make the
  validation set either.

**What isn't (vendor, through CSIDN or sales):**
1. May a commercial add-on send model data read through OAPI to an outside
   service?
2. Does §7 cover analysis results? It matters for a partner engineer
   sharing results with us for validation.
3. May results from a licensed engineer's runs be used to validate, or
   train, a model that is then sold?

The partner's models also usually belong to *their clients*. Validation
should use models the partner owns, or ones built for the purpose.

## Datasets

Ground motions, FEMA P-58, Hazus and the published studies are covered in
[sources.md](sources.md), with each one's terms.

## Who to ask, when it matters

| Question | Ask | Before |
|---|---|---|
| Selling a model trained on OpenSees outputs | UC Berkeley IPIRA/OTL, and Dr. Minjie Zhu | Any sale |
| "Internal purposes" for a company | A lawyer | Any sale |
| ETABS add-on sending data out; §7 and results | CSI (CSIDN or sales) | Building the add-on |
| GEM data for commercial use | licensing@globalquakemodel.org | Using any NC product |
