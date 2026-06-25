# Prompt matnlari

Barcha agent prompt matnlari shu yerda `.txt` fayllar sifatida saqlanadi —
Python kodida emas. Promptni tahrirlash uchun faqat tegishli faylni o'zgartiring,
kodga tegish shart emas. Fayllar `loader.load_prompt("<relpath>")` orqali yuklanadi.

```
texts/
├── dispatcher/
│   └── core_identity.txt          # Main agent (dispatcher) personasi → system.py
├── shared/
│   ├── autonomy.txt               # "execute, don't ask" — har agentга qo'shiladi
│   └── narrate_rule.txt           # sub-agent narrate/batch qoidalari
└── subagents/
    ├── flow_builder.txt           # flow-builder yo'riqnomasi (asosiy)
    ├── flow_builder_examples.txt  # to'liq bot misollari (dinamik kengaytiriladi)
    ├── crm_manager.txt
    ├── cms_manager.txt
    └── deploy_manager.txt
```

## Qaysi fayl qayerda ishlatiladi

| Fayl | Yuklovchi | Tegishli spec |
|------|-----------|---------------|
| `dispatcher/core_identity.txt` | `system.py` → `CORE_IDENTITY` | Main dispatcher |
| `shared/autonomy.txt` | `system.py` + `subagent_specs.py` | Barcha agentlar |
| `shared/narrate_rule.txt` | `subagent_specs.py` → `NARRATE_RULE` | Barcha sub-agentlar |
| `subagents/flow_builder.txt` + `_examples.txt` | `subagent_specs.py` → `FLOW_BUILDER` | flow-builder |
| `subagents/crm_manager.txt` | `subagent_specs.py` → `CRM_MANAGER` | crm-manager |
| `subagents/cms_manager.txt` | `subagent_specs.py` → `CMS_MANAGER` | cms-manager |
| `subagents/deploy_manager.txt` | `subagent_specs.py` → `DEPLOY_MANAGER` | deploy-manager |

## Maxsus belgilar

- `{node_params_table}` — `flow_builder.txt` ichida runtime'da node parametrlar
  jadvali bilan almashtiriladi (`render_summary_table()`). Boshqa placeholder yo'q.

## Model-ga xos qo'shimchalar

`model_addenda/<model-id-substring>.txt` — `system.py` configured executor model
uchun qo'shimcha yo'riqnoma yuklaydi (ixtiyoriy).
