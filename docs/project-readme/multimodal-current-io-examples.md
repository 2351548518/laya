# 按模态与输出类型查看真实样例

每个有数据的「模态 × 输出类型」组合各取三条 v6 训练记录，共 30 条。代码块展示原始请求与监督目标。媒体路径用示意文件名替换，记录 ID 和行号可在原始 `native/train.jsonl` 中查找。音频参考样例取自 Gemma 训练集。

[主说明](multimodal-current.md) · [64 类细任务样例](multimodal-current-examples.md) · [样例索引](assets/examples-index.json)

## 文本

### 文本 · `choice` 单选

**样例 1** · `tasksource:toxic-chat/toxicchat0124/toxicity` · `tasksource:toxic-chat-toxicchat0124-toxicity-259fcfe274:train:4210` · 训练集第 1,425,303 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "what is a makefile?",
    "questions": {
      "decision": {
        "type": "choice",
        "instructions": "Is this user prompt toxic?",
        "criteria": {
          "0": "not toxic",
          "1": "toxic"
        }
      }
    }
  },
  "expected": {
    "decision": "0"
  }
}
```

**样例 2** · `tasksource:glue/sst2` · `tasksource:glue-sst2-a91549683f:train:15149` · 训练集第 1,384,423 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "manipulative and ",
    "questions": {
      "decision": {
        "type": "choice",
        "instructions": "What sentiment does the text express?",
        "criteria": {
          "0": "negative",
          "1": "positive"
        }
      }
    }
  },
  "expected": {
    "decision": "0"
  }
}
```

**样例 3** · `tasksource:financial_phrasebank/sentences_allagree` · `tasksource:financial-phrasebank-sentences-allagree-fb77247656:train:511` · 训练集第 1,383,678 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "Welcome !",
    "questions": {
      "decision": {
        "type": "choice",
        "instructions": "What sentiment does the text express?",
        "criteria": {
          "0": "negative",
          "1": "neutral",
          "2": "positive"
        }
      }
    }
  },
  "expected": {
    "decision": "1"
  }
}
```

### 文本 · `noul` 是非判断

**样例 1** · `tasksource:go_emotions/simplified` · `tasksource:go-emotions-simplified-c563c5a238:train:9616:noul-label-verification` · 训练集第 1,387,589 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "You are ok.",
    "questions": {
      "decision": {
        "type": "noul",
        "instructions": "Is \"grief\" the correct label for this example?"
      }
    }
  },
  "expected": {
    "decision": false
  }
}
```

**样例 2** · `tasksource:clinc_oos/plus` · `tasksource:clinc-oos-plus-1b9b3d1a5a:train:15203:noul-label-verification` · 训练集第 1,379,513 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "what do rabbits eat",
    "questions": {
      "decision": {
        "type": "noul",
        "instructions": "Is \"oos\" the correct label for this example?"
      }
    }
  },
  "expected": {
    "decision": true
  }
}
```

**样例 3** · `tasksource:dynahate` · `tasksource:dynahate-224d6896b5:train:1664:noul-label-verification` · 训练集第 1,382,218 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "Black men are ugly",
    "questions": {
      "decision": {
        "type": "noul",
        "instructions": "Is \"hate\" the correct label for this example?"
      }
    }
  },
  "expected": {
    "decision": true
  }
}
```

### 文本 · `score` 评分

**样例 1** · `tasksource:multilingual/AfriSenti-twitter-sentiment/twi` · `tasksource:multilingual-AfriSenti-twitter-sentiment-twi-32a4496e5c:train:1486` · 训练集第 554,444 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "ayɛ mpo aka whonna bɛyɛ",
    "questions": {
      "decision": {
        "type": "score",
        "instructions": "What sentiment does the text express?",
        "criteria": [
          "negative",
          "neutral",
          "positive"
        ]
      }
    }
  },
  "gold": {
    "decision": {
      "probabilities": {
        "0": 0.0,
        "1": 0.5,
        "2": 0.5
      }
    }
  },
  "option_orders": {
    "decision": [
      0,
      1,
      2
    ]
  }
}
```

**样例 2** · `tasksource:multilingual/tweet_sentiment_multilingual` · `tasksource:multilingual-tweet-sentiment-multilingual-31edbb5b11:train:8470` · 训练集第 1,396,432 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "Hurr hurr ",
    "questions": {
      "decision": {
        "type": "score",
        "instructions": "What sentiment does the text express?",
        "criteria": [
          "negative",
          "neutral",
          "positive"
        ]
      }
    }
  },
  "expected": {
    "decision": 1
  }
}
```

**样例 3** · `tasksource:multilingual/AfriSenti-twitter-sentiment/ary` · `tasksource:multilingual-AfriSenti-twitter-sentiment-ary-4c2d98e466:train:5541` · 训练集第 1,444,840 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "مبروك يا أبا مكة'",
    "questions": {
      "decision": {
        "type": "score",
        "instructions": "What sentiment does the text express?",
        "criteria": [
          "negative",
          "neutral",
          "positive"
        ]
      }
    }
  },
  "expected": {
    "decision": 2
  }
}
```

## 图像

### 图像 · `choice` 单选

**样例 1** · `Cognitive-Lab/NayanaOCR_Corpus_2025:extractive-mcq` · `nayana:zh:document_10026_page_41:4` · 训练集第 1,032,798 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": [
      "__laya_media_image_0__",
      ""
    ],
    "questions": {
      "decision": {
        "type": "choice",
        "instructions": "NAIRU代表什么？",
        "criteria": {
          "A": "失业的非加速通货膨胀率",
          "B": "全国平均失业率",
          "C": "失业的不调整通货膨胀率",
          "D": "国家加速失业率"
        }
      }
    },
    "assets": {
      "__laya_media_image_0__": {
        "type": "image",
        "part": "__laya_media_image_0__"
      }
    }
  },
  "expected": {
    "decision": "A"
  },
  "media_files": {
    "__laya_media_image_0__": "media/example-1.jpg"
  }
}
```

**样例 2** · `HuggingFaceM4/A-OKVQA:native-mc` · `aokvqa-native:hjMqaofRj93ndSRibZqeZq` · 训练集第 355,446 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": [
      "__laya_media_image_0__",
      ""
    ],
    "questions": {
      "decision": {
        "type": "choice",
        "instructions": "What is the cat sleeping on?",
        "criteria": {
          "0": "couch",
          "1": "outdoors",
          "2": "floor",
          "3": "bed"
        }
      }
    },
    "assets": {
      "__laya_media_image_0__": {
        "type": "image",
        "part": "__laya_media_image_0__"
      }
    }
  },
  "expected": {
    "decision": "3"
  },
  "media_files": {
    "__laya_media_image_0__": "media/example-1.jpg"
  }
}
```

**样例 3** · `derek-thomas/ScienceQA:native-image` · `scienceqa-native:a0eaea2b9af55e19bd8bfb41cdf29ad2a30f63d6ce6b4e414f473bab78120eca` · 训练集第 1,349,262 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": [
      "__laya_media_image_0__",
      ""
    ],
    "questions": {
      "decision": {
        "type": "choice",
        "instructions": "Is a rock a solid, a liquid, or a gas?",
        "criteria": {
          "0": "a gas",
          "1": "a solid",
          "2": "a liquid"
        }
      }
    },
    "assets": {
      "__laya_media_image_0__": {
        "type": "image",
        "part": "__laya_media_image_0__"
      }
    }
  },
  "expected": {
    "decision": "1"
  },
  "media_files": {
    "__laya_media_image_0__": "media/example-1.png"
  }
}
```

### 图像 · `noul` 是非判断

**样例 1** · `lmms-lab-encoder/VQAv2:unanimous-native-binary` · `vqa2-unanimous:525101015` · 训练集第 1,321,141 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": [
      "__laya_media_image_0__",
      ""
    ],
    "questions": {
      "decision": {
        "type": "noul",
        "instructions": "Is this a toy?"
      }
    },
    "assets": {
      "__laya_media_image_0__": {
        "type": "image",
        "part": "__laya_media_image_0__"
      }
    }
  },
  "expected": {
    "decision": false
  },
  "media_files": {
    "__laya_media_image_0__": "media/example-1.img"
  }
}
```

**样例 2** · `VQA-v2:official-train-consensus` · `vqa2-train:384740002` · 训练集第 1,245,713 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": [
      "__laya_media_image_0__",
      ""
    ],
    "questions": {
      "decision": {
        "type": "noul",
        "instructions": "Is that healthy?"
      }
    },
    "assets": {
      "__laya_media_image_0__": {
        "type": "image",
        "part": "__laya_media_image_0__"
      }
    }
  },
  "expected": {
    "decision": false
  },
  "media_files": {
    "__laya_media_image_0__": "media/example-1.jpg"
  }
}
```

**样例 3** · `docling-project/DocLayNet-v1.1` · `doclaynet:79f6a44c875a6992f27f85cb65b6cdc48c45a8ccc5c67c4bd3521a4bdf685ad8:presence:table` · 训练集第 1,203,886 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": [
      "__laya_media_image_0__",
      ""
    ],
    "questions": {
      "decision": {
        "type": "noul",
        "instructions": "Does this page contain a table?"
      }
    },
    "assets": {
      "__laya_media_image_0__": {
        "type": "image",
        "part": "__laya_media_image_0__"
      }
    }
  },
  "expected": {
    "decision": true
  },
  "media_files": {
    "__laya_media_image_0__": "media/example-1.jpg"
  }
}
```

### 图像 · `score` 评分

**样例 1** · `koniq` · `koniq:5602815566.jpg` · 训练集第 507,874 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": [
      "__laya_media_image_0__",
      ""
    ],
    "questions": {
      "decision": {
        "type": "score",
        "instructions": "Rate the technical visual quality of this photograph.",
        "criteria": [
          "bad",
          "poor",
          "fair",
          "good",
          "excellent"
        ]
      }
    },
    "assets": {
      "__laya_media_image_0__": {
        "type": "image",
        "part": "__laya_media_image_0__"
      }
    }
  },
  "gold": {
    "decision": {
      "probabilities": {
        "0": 0.0,
        "1": 0.07,
        "2": 0.51,
        "3": 0.37,
        "4": 0.05
      }
    }
  },
  "option_orders": {
    "decision": [
      0,
      1,
      4,
      2,
      3
    ]
  },
  "media_files": {
    "__laya_media_image_0__": "media/example-1.jpg"
  }
}
```

**样例 2** · `tasksource:vision/met-meme/offensiveness` · `tasksource:vision/met-meme/offensiveness:train:040f969d61e6b54f7aaac5f2:437004ba3497a904:2fd8b6d4cd85060123e8b917` · 训练集第 913,871 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": [
      "__laya_media_image_0__",
      "JUST ADD WATER!"
    ],
    "questions": {
      "decision": {
        "type": "score",
        "instructions": "How offensive is this meme?",
        "criteria": [
          "non-offensive",
          "slightly offensive",
          "moderately offensive",
          "very offensive"
        ]
      }
    },
    "assets": {
      "__laya_media_image_0__": {
        "type": "image",
        "part": "__laya_media_image_0__"
      }
    }
  },
  "expected": {
    "decision": 0
  },
  "option_orders": {
    "decision": [
      1,
      0,
      3,
      2
    ]
  },
  "media_files": {
    "__laya_media_image_0__": "media/example-1.png"
  }
}
```

**样例 3** · `Rapidata/human-coherence-preferences-images` · `rapidata-coherence:d3b0708116f8830c8f1eb6d8bcb13b90cfddfe0948f124dad1c4e7cf30e65ed5` · 训练集第 1,481,256 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": [
      "__laya_media_image_0__",
      "__laya_media_image_1__",
      ""
    ],
    "questions": {
      "decision": {
        "type": "score",
        "criteria": [
          "The first image is clearly more visually coherent.",
          "The first image is somewhat more visually coherent.",
          "The two images are close in visual coherence.",
          "The second image is somewhat more visually coherent.",
          "The second image is clearly more visually coherent."
        ],
        "instructions": "Compare the visual coherence of the two images in input order. Consider visual errors, unnatural shapes and glitches."
      }
    },
    "assets": {
      "__laya_media_image_0__": {
        "type": "image",
        "part": "__laya_media_image_0__"
      },
      "__laya_media_image_1__": {
        "type": "image",
        "part": "__laya_media_image_1__"
      }
    }
  },
  "expected": {
    "decision": 0
  },
  "media_files": {
    "__laya_media_image_0__": "media/example-1.jpg",
    "__laya_media_image_1__": "media/example-2.jpg"
  }
}
```

### 图像 · `box` 单框定位

**样例 1** · `jxu124/refcocoplus` · `17c5eb26b236dc2b41093891c83dd7cebfd5c23c7053776c43808a34a52619d7` · 训练集第 928,181 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": [
      "__laya_media_image_0__",
      ""
    ],
    "questions": {
      "ground": {
        "type": "box",
        "instructions": "black turban"
      }
    },
    "assets": {
      "__laya_media_image_0__": {
        "type": "image",
        "part": "__laya_media_image_0__"
      }
    }
  },
  "expected": {
    "ground": {
      "type": "box",
      "box": {
        "left": 0.731828125,
        "top": 0.10367058823529413,
        "right": 1.0,
        "bottom": 1.0
      },
      "confidence": null
    }
  },
  "media_files": {
    "__laya_media_image_0__": "media/example-1.jpg"
  }
}
```

**样例 2** · `coco2017` · `6cacf307bf1b5cb1a9c4dd3764766269f6a1872f8529eef5e6d43e46dee5621f` · 训练集第 668,985 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": [
      "__laya_media_image_0__",
      ""
    ],
    "questions": {
      "ground": {
        "type": "box",
        "instructions": "couch"
      }
    },
    "assets": {
      "__laya_media_image_0__": {
        "type": "image",
        "part": "__laya_media_image_0__"
      }
    }
  },
  "expected": {
    "ground": {
      "type": "box",
      "box": {
        "left": 0.004296875,
        "top": 0.7139812646370024,
        "right": 1.0,
        "bottom": 0.9892505854800937
      },
      "confidence": null
    }
  },
  "media_files": {
    "__laya_media_image_0__": "media/example-1.jpg"
  }
}
```

**样例 3** · `jxu124/refcoco` · `278a1c49f2297ea80280dfa0b526e4e7b71c9922be663da60e0b88c9c6e1c138` · 训练集第 589,273 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": [
      "__laya_media_image_0__",
      ""
    ],
    "questions": {
      "ground": {
        "type": "box",
        "instructions": "left e"
      }
    },
    "assets": {
      "__laya_media_image_0__": {
        "type": "image",
        "part": "__laya_media_image_0__"
      }
    }
  },
  "expected": {
    "ground": {
      "type": "box",
      "box": {
        "left": 0.0389375,
        "top": 0.32899999999999996,
        "right": 0.428328125,
        "bottom": 0.7398333333333333
      },
      "confidence": null
    }
  },
  "media_files": {
    "__laya_media_image_0__": "media/example-1.jpg"
  }
}
```

## 视频

### 视频 · `choice` 单选

**样例 1** · `CLEVRER` · `clevrer:video_09884:descriptive:0` · 训练集第 11,023 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "",
    "questions": {
      "decision": {
        "criteria": {
          "A": "metal",
          "B": "rubber"
        },
        "instructions": "What is the material of the object to collide with the red cube?",
        "type": "choice"
      }
    },
    "video": {
      "part": "__laya_media_video_0__"
    },
    "video_audio": false,
    "video_frames": 8
  },
  "expected": {
    "decision": "A"
  },
  "media_files": {
    "__laya_media_video_0__": "media/example-1.mp4"
  }
}
```

**样例 2** · `nextqa` · `nextqa:3536560404:8` · 训练集第 608,171 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "",
    "questions": {
      "decision": {
        "type": "choice",
        "instructions": "what did the man do after he approached the kitchen counter?",
        "criteria": {
          "2": "eating",
          "4": "looked out",
          "1": "pick up paper",
          "3": "look at him",
          "0": "dance"
        }
      }
    },
    "video": {
      "part": "__laya_media_video_0__"
    },
    "video_audio": false,
    "video_frames": 8
  },
  "expected": {
    "decision": "1"
  },
  "option_orders": {
    "decision": [
      2,
      4,
      1,
      3,
      0
    ]
  },
  "media_files": {
    "__laya_media_video_0__": "media/example-1.mp4"
  }
}
```

**样例 3** · `CLEVRER` · `clevrer:video_06890:descriptive:6` · 训练集第 14,219 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "",
    "questions": {
      "decision": {
        "criteria": {
          "A": "metal",
          "B": "rubber"
        },
        "instructions": "What material is the stationary gray object when the video ends?",
        "type": "choice"
      }
    },
    "video": {
      "part": "__laya_media_video_0__"
    },
    "video_audio": false,
    "video_frames": 8
  },
  "expected": {
    "decision": "A"
  },
  "media_files": {
    "__laya_media_video_0__": "media/example-1.mp4"
  }
}
```

### 视频 · `noul` 是非判断

**样例 1** · `ssv2:native-action-verification` · `v5-ssv2:166017:verify:0edbab6b16d59191` · 训练集第 1,464,800 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "",
    "questions": {
      "decision": {
        "type": "noul",
        "instructions": "Is the primary action in the video described by this phrase? holding something"
      }
    },
    "video": {
      "part": "__laya_media_video_0__"
    },
    "video_audio": false,
    "video_frames": 4
  },
  "expected": {
    "decision": true
  },
  "media_files": {
    "__laya_media_video_0__": "media/example-1.webm"
  }
}
```

**样例 2** · `CLEVRER` · `clevrer:video_07868:descriptive:10` · 训练集第 748,653 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "",
    "questions": {
      "decision": {
        "instructions": "Are there any moving metal objects when the video ends?",
        "type": "noul"
      }
    },
    "video": {
      "part": "__laya_media_video_0__"
    },
    "video_audio": false,
    "video_frames": 8
  },
  "expected": {
    "decision": true
  },
  "option_orders": {
    "decision": [
      1,
      0
    ]
  },
  "media_files": {
    "__laya_media_video_0__": "media/example-1.mp4"
  }
}
```

**样例 3** · `ucf101-full` · `ucf101-full:v_PushUps_g08_c02.avi` · 训练集第 410,550 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "",
    "questions": {
      "decision": {
        "type": "noul",
        "instructions": "Which main action is performed in the clip? Is it push ups?"
      }
    },
    "video": {
      "part": "__laya_media_video_0__"
    },
    "video_audio": false,
    "video_frames": 4
  },
  "expected": {
    "decision": true
  },
  "option_orders": {
    "decision": [
      1,
      0
    ]
  },
  "media_files": {
    "__laya_media_video_0__": "media/example-1.avi"
  }
}
```

### 视频 · `score` 评分

**样例 1** · `TIGER-Lab/VideoFeedback:annotated` · `videofeedback:0003567:dynamic degree` · 训练集第 259,275 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "{\"generation_prompt\": \"Ally McBeal eating a pizza really fast  \"}",
    "questions": {
      "decision": {
        "type": "score",
        "instructions": "Assess dynamic degree (degree of dynamic change in the video) using the original VideoFeedback 1–4 scale. Judge the visible video; use the generation prompt where relevant.",
        "criteria": [
          "1: Bad",
          "2: Average",
          "3: Good",
          "4: Real or perfect"
        ]
      }
    },
    "video": {
      "part": "__laya_media_video_0__"
    },
    "video_audio": false,
    "video_frames": 16
  },
  "expected": {
    "decision": 2
  },
  "media_files": {
    "__laya_media_video_0__": "media/example-1.mp4"
  }
}
```

**样例 2** · `MMSP/KoNViD-1k:native-mos` · `konvid-mos:7700672030` · 训练集第 1,108,835 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "",
    "questions": {
      "decision": {
        "type": "score",
        "instructions": "Assess overall perceptual video quality on the original human 1-to-5 scale.",
        "criteria": [
          "1: Bad",
          "2: Poor",
          "3: Fair",
          "4: Good",
          "5: Excellent"
        ]
      }
    },
    "video": {
      "part": "__laya_media_video_0__"
    },
    "video_audio": false,
    "video_frames": 16
  },
  "expected": {
    "decision": 2.3
  },
  "media_files": {
    "__laya_media_video_0__": "media/example-1.mp4"
  }
}
```

**样例 3** · `teowu/DIVIDE-MaxWell:official-train-MOS` · `divide-maxwell:1042.mp4` · 训练集第 179,154 行。

```json
{
  "request": {
    "model": "multimodal",
    "state": "",
    "questions": {
      "decision": {
        "type": "score",
        "instructions": "Rate the overall visual quality of the video.",
        "criteria": [
          "0: Extremely poor overall visual quality (original MOS anchor 1)",
          "1: Poor overall visual quality (original MOS anchor 2)",
          "2: Fair overall visual quality (original MOS anchor 3)",
          "3: Good overall visual quality (original MOS anchor 4)",
          "4: Excellent overall visual quality (original MOS anchor 5)"
        ]
      }
    },
    "video": {
      "part": "__laya_media_video_0__"
    },
    "video_audio": false,
    "video_frames": 4
  },
  "expected": {
    "decision": 1.6
  },
  "media_files": {
    "__laya_media_video_0__": "media/example-1.mp4"
  }
}
```

<a name="gemma-audio"></a>

## Gemma 音频参考样例

下列九条来自本地 `open-jev-omni-gemma-filtered` 的训练分片，展示项目的音频输入能力。v6 的 Qwen3.5-4B 训练集音频记录数为 0。

### 音频 · 单选

- **audio-jev:next_speaker** · `audio-jev:rlcd-ami-ES2013b-0132ddccdc9d1770deea-next_speaker`。输入包含音频和文本 ``。问题 `Predict who will next begin speaking within 2 seconds after this audio ends. Number people by the order their…`。监督目标 `{"probabilities":{"new_speaker":0.0,"none":0.0,"speaker_1":1.0}}`。
- **inclusionAI/AudioMCQ** · `audiomcq-v5:7abdbf61f5d1f1bfaef98a4c5b8c987ede18284744d8b092afa336c07183d341`。输入包含音频和文本 ``。问题 `When does the sound end in the audio clip?`。监督目标 `{"probabilities":{"A":0.0,"B":0.0,"C":0.0,"D":1.0}}`。
- **speech-commands** · `speech-commands:0d6d7360:off:0d6d7360_nohash_4.wav`。输入包含音频和文本 ``。问题 `Which command word is spoken in this recording?`。监督目标 `{"probabilities":{"go":0.0,"left":0.0,"bed":0.0,"seven":0.0,"eight":0.0,"follow":0.0,"visual":0.0,"house":0.0,"learn":0.0,"two":0.0,"down":0.0,"cat":0.0,"off":1.0,"wow":0.0,"four":0.0,"happy":0.0,"marvin":0.0,"five":0.0,"right":0.0,"bird":0.0}}`。

### 音频 · 是非判断

- **audio-jev:speech_activity** · `audio-jev:rlcd-ami-ES2015b-f39a6b8f7c50a0593837-speech_activity`。输入包含音频和文本 ``。问题 `Does any person speak in the final 200 milliseconds of this audio?`。监督目标 `{"probabilities":{"false":1.0,"true":0.0}}`。
- **fsd50k** · `fsd50k:395407:dominant:/m/01j3sz`。输入包含音频和文本 ``。问题 `Is laughter a predominant sound in this recording?`。监督目标 `{"probabilities":{"false":0.0,"true":1.0}}`。
- **speech-commands** · `speech-commands:98170b49:dog:98170b49_nohash_0.wav`。输入包含音频和文本 ``。问题 `Which command word is spoken in this recording? Is it dog?`。监督目标 `{"probabilities":{"false":0.0,"true":1.0}}`。

### 音频 · 评分

- **audio-jev:speaker_count** · `audio-jev:rlcd-ami-ES2008b-9c095476614c0c1f288b-speaker_count`。输入包含音频和文本 ``。问题 `How many distinct people are audibly speaking anywhere in this audio window? Count each person once, including…`。监督目标 `{"probabilities":{"0":0.0,"1":0.0,"2":0.0,"3":0.0,"4":1.0,"5":0.0,"6":0.0,"7":0.0,"8":0.0,"9":0.0,"10":0.0}}`。
- **declare-lab/MELD** · `meld:dia10_utt1:sentiment`。输入包含音频和文本 `{"preceding_utterances": [{"speaker": "Phoebe", "utterance": "Nobody!"}], "current_ut…`。问题 `What sentiment does the current speaker express in this utterance?`。监督目标 `{"probabilities":{"0":1.0,"1":0.0,"2":0.0}}`。
- **fsd50k** · `fsd50k:406720:prominence:/m/07q7njn`。输入包含音频和文本 ``。问题 `Rate the audibility/prominence of chink and clink.`。监督目标 `{"probabilities":{"0":0.0,"1":0.0,"2":1.0}}`。
