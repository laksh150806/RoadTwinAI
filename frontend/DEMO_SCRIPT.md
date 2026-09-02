# RoadTwin AI — Demo Script (Member 4)

Runtime: ~90 seconds. Full screen the browser (F11) before starting.

---

**[Screen: idle state, R-1042 green, score 82]**

> "Road maintenance today is reactive. A pothole gets reported only after someone hits it.
> RoadTwin flips that — every drive becomes an observation, and the twin updates itself."

Point at map.

> "Here's our live digital twin of a road network. Green segments are stable, orange is
> watch-list. This one — R-1042 — looks fine right now. Condition score 82, priority #7."

**[Click "Stream New Dashcam Footage"]**

> "A dashcam just drove this stretch. Watch what happens with zero human input."

Narrate as the pipeline lights up left to right (~4 seconds total):

> "Vision agent detects the damage — three new potholes, four new cracks, standing water.
> Sensor fusion pins it to the exact GPS location using accelerometer data.
> The condition model recomputes the score in real time — 82 drops to 64.
> The deterioration model forecasts forward: 47 in thirty days, 31 in sixty — this road
> fails if nobody touches it."

**[Score/priority flip red, heatmap flares]**

> "Risk flips from low to high. Priority jumps from #7 to #1 — automatically ranked against
> every other segment in the network, not just flagged in isolation."

**[Toast appears: "Authorities alerted automatically"]**

> "The decision engine doesn't just report the damage — it alerts the authority and
> recommends a repair window, 11 PM to 5 AM, when it'll disrupt traffic least."

**[Beat, gesture at whole screen]**

> "This loop — new footage in, updated priority out — is the entire product. It's
> predictive, not reactive, and it runs on hardware every authority already has:
> a dashcam and a phone."

---

### Fallback lines (if asked)
- **"Why not just pothole detection?"** → "Detection is the input, not the product. The
  product is the twin that keeps forecasting and re-ranking as new data comes in."
- **"Is this live data?"** → "This build runs on a scripted scenario for the demo; the
  vision, fusion, and decision agents are built separately by the team and plug into this
  same twin view through the data contract in `data-contract.json`."
- **"What's next?"** → "Wire the real Vision/Fusion/Decision agents into this dashboard,
  and extend the what-if traffic simulation shown on slide 07."
