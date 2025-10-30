# experts_input.py


experts = [
    {
        "expert_id": "E1",
        "data": [
            {"alternative": "EGT",
             "tolerance": (1,3,5), "interest": (3,5,7),
             "bus_new": (0.2,0.3,0.3), "instrument_complexity": (0.5,0.6,0.6),
             "risk": (4.2,6.2,8.0)},
            {"alternative": "KI",
             "tolerance": (5,7,9), "interest": (3,5,7),
             "bus_new": (0.2,0.3,0.3), "instrument_complexity": (0.5,0.6,0.6),
             "risk": (3.6,5.6,7.5)},
            {"alternative": "IBS",
             "tolerance": (3,5,7), "interest": (5,7,9),
             "bus_new": (0.2,0.3,0.3), "instrument_complexity": (0.5,0.6,0.6),
             "risk": (3.7,5.6,7.4)},
            {"alternative": "LA",
             "tolerance": (1,3,5), "interest": (5,7,9),
             "bus_new": (0.2,0.3,0.3), "instrument_complexity": (0.5,0.6,0.6),
             "risk": (4.2,6.2,8.0)},
        ],
        # mantener índices numéricos (misma convención). cost ocupa el lugar de maturity
        "criteria_order": [1, 4, 2, 3, 5, 6],
        "chis": [ (1,1.5,2), (1.5,2.25,3), (1,1,1), (1,1.25,1.5), (2,2.5,3) ]
    },
    {
        "expert_id": "E2",
        "data": [
            {"alternative": "EGT",
             "tolerance": (2,4,6), "interest": (2,4,6),
             "bus_new": (0.2,0.3,0.3), "instrument_complexity": (0.5,0.6,0.6),
             "risk": (3.9,5.5,7.0)},
            {"alternative": "KI",
             "tolerance": (4,6,8), "interest": (2,4,6),
             "bus_new": (0.2,0.3,0.3), "instrument_complexity": (0.5,0.6,0.6),
             "risk": (3.2,5.0,7.1)},
            {"alternative": "IBS",
             "tolerance": (2,4,6), "interest": (4,6,8),
             "bus_new": (0.2,0.3,0.3), "instrument_complexity": (0.5,0.6,0.6),
             "risk": (3.5,5.2,7.0)},
            {"alternative": "LA",
             "tolerance": (1,3,5), "interest": (4,6,8),
             "bus_new": (0.2,0.3,0.3), "instrument_complexity": (0.5,0.6,0.6),
             "risk": (4.0,6.0,8.2)},
        ],
        "criteria_order": [1, 4, 2, 3, 5, 6],
        "chis": [ (1,1.25,1.5), (1,1.5,2), (1,1,1), (1.25,1.6,2), (1.8,2.2,2.8) ]
    }
]


