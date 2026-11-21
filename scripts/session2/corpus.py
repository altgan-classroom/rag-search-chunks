"""Invented policy corpus for Session 2 (nothing real, nothing confidential). Each doc is markdown text. The escape-of-water clause in the 2026 home
policy is deliberately longer than the 500-character fixed window used in the chunking demo, so a fixed-window cut must fall inside it."""

DOCS = {}

DOCS["HomePolicy_HP-2026-0417"] = """# Home Insurance Policy HP-2026-0417

Issued by Northgate Insurance. Effective 1 January 2026. Policyholder: M. Rao, 14 Lake View Road.

## 1. Who is covered

The policy covers the named policyholder and the members of the household who live at the insured address. Visitors are not covered for their own belongings. Cover applies to the main dwelling, outbuildings and fixed fittings.

## 2. Summary of limits

| Cover | Limit per claim | Deductible |
|---|---|---|
| Buildings | $450,000 | $500 |
| Contents | $60,000 | $500 |
| Escape of water | $10,000 | $750 |
| Theft | $20,000 | $1,000 |
| Alternative accommodation | $15,000 | none |

## 3. Escape of water

We cover damage to the buildings and contents caused by the sudden escape of water from a burst pipe, a tank or a fixed appliance, up to the escape of water limit shown in section 2, which is $10,000 per claim. This cover does not apply where the property was left unheated for more than 48 consecutive hours between November and March, or where the damage results from a lack of reasonable maintenance, for example a pipe that was known to be corroded. You must take reasonable steps to prevent further damage and you must report the loss within 14 days of discovering it.

## 4. Theft

We cover theft or attempted theft where there is evidence of forced entry. Cover for items kept outside the dwelling is limited to $2,000 in total.

## 5. General exclusions

We do not cover wear and tear, gradual deterioration, damage caused deliberately by the policyholder, or loss from war, nuclear risks or flood from a river or the sea.
"""

DOCS["HomePolicy_HP-2025-0391"] = """# Home Insurance Policy HP-2025-0391

Issued by Northgate Insurance. Effective 1 January 2025. Superseded by the 2026 policy wording.

## 1. Who is covered

The policy covers the named policyholder and the members of the household who live at the insured address. Cover applies to the main dwelling and fixed fittings.

## 2. Summary of limits

| Cover | Limit per claim | Deductible |
|---|---|---|
| Buildings | $400,000 | $500 |
| Contents | $50,000 | $500 |
| Escape of water | $7,500 | $500 |
| Theft | $15,000 | $1,000 |

## 3. Escape of water

We cover damage to the buildings and contents caused by the sudden escape of water from a burst pipe, a tank or a fixed appliance, up to the escape of water limit shown in section 2, which is $7,500 per claim. This cover does not apply where the property was left unheated for more than 72 consecutive hours between November and March, or where the damage results from a lack of reasonable maintenance. You must report the loss within 30 days of discovering it.

## 4. General exclusions

We do not cover wear and tear, gradual deterioration, damage caused deliberately by the policyholder, or flood from a river or the sea.
"""

DOCS["MotorPolicy_MP-2026-1180"] = """# Motor Insurance Policy MP-2026-1180

Issued by Northgate Insurance. Effective 1 March 2026. Vehicle: hatchback, registration KA 05 MN 4821.

## 1. What is covered

Comprehensive cover applies to accidental damage, fire, theft and third party liability. Windscreen and glass damage is covered with a reduced excess.

## 2. Excess and limits

| Cover | Limit | Excess |
|---|---|---|
| Accidental damage | Market value | $400 |
| Windscreen and glass | $1,200 | $50 |
| Theft of vehicle | Market value | $400 |
| Third party property | $1,000,000 | none |
| Courtesy car | 14 days | none |

## 3. Claims

Report any accident within 48 hours. Do not admit liability. Keep receipts for any emergency repairs and take photographs of the damage and the other vehicle.

## 4. Exclusions

We do not cover mechanical breakdown, tyres damaged by wear, or driving without a valid licence. Cover is not available for racing or competitive events.
"""

DOCS["TravelPolicy_TP-2026-0233"] = """# Travel Insurance Policy TP-2026-0233

Issued by Northgate Insurance. Effective 1 April 2026. Single trip, two travellers.

## 1. What is covered

The policy covers emergency medical treatment abroad up to $250,000, lost baggage up to $2,000 and trip cancellation up to $5,000 when the reason is listed in section 2.

## 2. Cancellation

You can claim for cancellation caused by serious illness, injury, bereavement or jury service. Cancellation because you changed your mind is not covered. Cancellation because of a pandemic is not covered unless the traveller is hospitalised.

## 3. Natural events

Damage to belongings caused by flood, storm or earthquake is not covered under this policy. Medical costs caused by these events remain covered.

## 4. Claims

Contact the claims line within 30 days of returning home. Claim code TR-7731 applies to trip cancellation claims.
"""

DOCS["ClaimsGuide_2026"] = """# Claims Guide 2026

This guide explains how to make a claim with Northgate Insurance.

## 1. Before you claim

Make the property safe and stop any water leak if you can do so safely. Take photographs of the damage before you clean up. Keep damaged items until the loss adjuster has seen them.

## 2. Claim codes

Use claim code CL-4471 for water damage and escape of water claims. Use claim code CL-2290 for theft. Use claim code CL-3015 for glass and windscreen claims. Quote the code and your policy number when you call.

## 3. What happens next

A claims handler contacts you within two working days. For water damage a loss adjuster may visit the property. Payments are made within ten working days of agreeing the claim.

## 4. Complaints

If you are unhappy with the handling of a claim you can ask for a review by a senior handler.
"""

DOCS["LandlordPolicy_LP-2026-0850"] = """# Landlord Insurance Policy LP-2026-0850

Issued by Northgate Insurance. Effective 1 February 2026. Rental property at 7 Park Lane.

## 1. What is covered

The policy covers the building, landlord fixtures and loss of rent when the property cannot be lived in after an insured event.

## 2. Limits

| Cover | Limit per claim | Deductible |
|---|---|---|
| Buildings | $300,000 | $750 |
| Loss of rent | $24,000 | none |
| Escape of water | $12,000 | $1,000 |

## 3. Escape of water

We cover damage from the sudden escape of water from a burst pipe or tank up to $12,000 per claim. Damage caused by a tenant who left taps running is covered only if the tenant was in occupation under a written tenancy agreement.

## 4. Tenants

Tenants are not covered for their own belongings under this policy.
"""

# (id, question, kind, gold markers that must ALL appear in the retrieved text to answer correctly, note)
QUESTIONS = [
    ("q1", "Is water damage from a burst pipe covered?", "running bug", ["$10,000 per claim", "48 consecutive hours"], "Needs the limit AND the exclusion from the 2026 home policy"),
    ("q2", "Can I get my money back if my pipes freeze and flood the house while we are away?", "paraphrase", ["unheated", "48 consecutive hours"], "No shared keywords with the clause; the answer is the exclusion"),
    ("q3", "What is the deductible for escape of water under policy HP-2026-0417?", "exact ID", ["HP-2026-0417", "$750"], "Exact policy number"),
    ("q4", "claim code CL-4471", "exact code", ["CL-4471"], "Exact claim code"),
    ("q5", "What is the escape of water limit in the 2025 home policy?", "version", ["$7,500"], "Two versions of the same policy exist"),
    ("q6", "How much is the excess for a windscreen claim on my motor policy?", "table lookup", ["Windscreen", "$50"], "Value sits in a table"),
    ("q7", "Is flood damage to my belongings covered when I travel?", "negation", ["not covered"], "The answer is a 'not covered' statement"),
    ("q8", "What happens if I cancel my trip because I changed my mind?", "negation", ["changed your mind", "not covered"], "Answer is a 'not covered' statement"),
    ("q9", "Within how many days must I report an escape of water loss?", "detail in clause", ["14 days"], "A detail from the end of the clause"),
    ("q10", "Does my landlord policy cover a tenant's belongings?", "negation", ["not covered for their own belongings"], "A short 'not covered' line"),
    ("q11", "Will you reimburse me if my plumbing ruptures and soaks the carpet in my house?", "no shared words", ["$10,000 per claim"], "Paraphrase with almost no keywords from the clause"),
    ("q12", "Am I protected if ice damages my pipes when nobody is home for a few days?", "no shared words", ["48 consecutive hours"], "Paraphrase; the answer is the exclusion"),
    ("q13", "What do I pay out of pocket before the insurer pays for a smashed car window?", "no shared words", ["$1,200 | $50"], "Asks about the excess without using the word"),
]

GOLD = {  # question id -> chunks that answer it: (doc, text that the chunk must contain). Any one counts.
    "q1": [("HomePolicy_HP-2026-0417", "48 consecutive hours")],
    "q2": [("HomePolicy_HP-2026-0417", "48 consecutive hours")],
    "q3": [("HomePolicy_HP-2026-0417", "| Escape of water | $10,000 | $750 |")],
    "q4": [("ClaimsGuide_2026", "CL-4471")],
    "q5": [("HomePolicy_HP-2025-0391", "$7,500")],
    "q6": [("MotorPolicy_MP-2026-1180", "Windscreen and glass | $1,200 | $50")],
    "q7": [("TravelPolicy_TP-2026-0233", "flood, storm or earthquake")],
    "q8": [("TravelPolicy_TP-2026-0233", "changed your mind")],
    "q9": [("HomePolicy_HP-2026-0417", "14 days")],
    "q10": [("LandlordPolicy_LP-2026-0850", "Tenants are not covered")],
    "q11": [("HomePolicy_HP-2026-0417", "48 consecutive hours")],
    "q12": [("HomePolicy_HP-2026-0417", "48 consecutive hours")],
    "q13": [("MotorPolicy_MP-2026-1180", "Windscreen and glass | $1,200 | $50")],
}

SENTENCES = [
    ("paraphrase", "A pipe burst and flooded the kitchen."),
    ("paraphrase", "Water escaped from a ruptured pipe and covered the kitchen floor."),
    ("paraphrase", "The kitchen was flooded because of a broken water pipe."),
    ("negation", "Water damage from a burst pipe is covered."),
    ("negation", "Water damage from a burst pipe is not covered."),
    ("negation", "Damage from a burst pipe is excluded from the policy."),
    ("numbers", "The escape of water limit is $10,000 per claim."),
    ("numbers", "The escape of water limit is $1,000 per claim."),
    ("numbers", "The theft limit is $20,000 per claim."),
    ("ids", "Policy HP-2026-0417 covers the main dwelling."),
    ("ids", "Policy HP-2026-0471 covers the main dwelling."),
    ("ids", "Policy MP-2026-1180 covers the vehicle."),
    ("ids", "Claim code CL-4471 is used for water damage."),
    ("ids", "Claim code CL-4417 is used for water damage."),
    ("insurance", "Report any accident within 48 hours."),
    ("insurance", "Windscreen damage has a reduced excess of $50."),
    ("insurance", "Tenants are not covered for their own belongings."),
    ("insurance", "Cancellation because you changed your mind is not covered."),
    ("insurance", "How do I make a claim after a car accident?"),
    ("insurance", "We will pay for damage caused by a sudden escape of water."),
    ("insurance", "Contact the claims line within 30 days of returning home."),
    ("unrelated", "A recipe for lentil soup needs onions and cumin."),
    ("unrelated", "The cat slept on the warm windowsill."),
    ("unrelated", "What is the weather forecast for tomorrow?"),
]
PROBES = ["Is a burst pipe covered?", "What is the limit for water damage?", "Which claim code do I use for a leak?", "My pipe broke and the floor is wet.", "Do I have to pay anything first?"]
