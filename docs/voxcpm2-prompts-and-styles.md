# VoxCPM2

**Tokenizer-Free TTS for Multilingual Speech Generation, Creative Voice Design, and True-to-Life Cloning**

VoxCPM Team — ModelBest | THUHCSI | OpenBMB

> Abstract: VoxCPM2 is a tokenizer-free Text-to-Speech system built on an end-to-end diffusion autoregressive architecture (LocEnc → TSLM → RALM → LocDiT), bypassing discrete tokenization to achieve highly natural and expressive synthesis. Based on a MiniCPM-4 backbone with 2B parameters and trained on over 2 million hours of multilingual speech data, VoxCPM2 supports 30 languages and outputs 48kHz studio-quality audio.

## Key Features

- **30-Language Multilingual** — No language tag needed; synthesize in any of the 30 supported languages directly
- **Voice Design** — Generate a novel voice from a natural-language description (gender, age, tone, emotion, pace); no reference audio required
- **Controllable Cloning** — Clone any voice from a short clip, with optional style guidance to steer emotion, pace, and expression while preserving timbre
- **48kHz Studio-Quality Output** — Accepts 16kHz reference; outputs 48kHz via AudioVAE V2's built-in super-resolution

---

## Multilingual: 30-Language & Chinese Dialect Support

| Language | Target Text |
|:---|:---|
| English | Researchers announced today that a new deep-sea species has been discovered off the coast of Norway, marking one of the most significant marine biology findings in the past decade. The creature, roughly the size of a human hand, emits a faint bioluminescent glow in the dark waters of the Atlantic. |
| Chinese | 各位听众朋友大家好，欢迎收听今天的科技前沿节目。今天我们要聊的话题是人工智能语音合成技术的最新进展，以及它将如何改变我们日常生活中的交互方式。 |
| Japanese | 春になると、京都の哲学の道は桜のトンネルに変わります。花びらが風に舞い、水面にそっと落ちてゆく。地元の人々は毎年この光景を見ても、決して飽きることはないと言います。日本の春は、ただの季節ではなく、心を洗う儀式のようなものです。 |
| Korean | 서울의 밤은 언제나 활기가 넘칩니다. 명동 거리에는 다양한 음식 냄새가 가득하고, 한강 위로는 형형색색의 불빛이 물결 위에 춤을 춥니다. 이 도시는 전통과 현대가 자연스럽게 어우러지는 곳으로, 방문하는 모든 사람에게 특별한 경험을 선사합니다. |
| French | La cuisine française est bien plus qu'un simple repas — c'est un art de vivre. Du croissant doré du petit-déjeuner au dîner aux chandelles, chaque plat raconte une histoire de terroir et de savoir-faire transmis de génération en génération. Il n'est pas surprenant que la gastronomie française soit inscrite au patrimoine immatériel de l'humanité. |
| Spanish | En las calles de Barcelona, la arquitectura de Gaudí transforma la ciudad en un museo al aire libre. La Sagrada Familia se alza hacia el cielo como un himno de piedra, mientras las fachadas ondulantes de la Casa Batlló parecen cobrar vida bajo la luz del Mediterráneo. Cada rincón de esta ciudad cuenta una historia diferente. |
| German | Die Alpen erstrecken sich über acht Länder und bilden das Rückgrat Europas. Jedes Jahr ziehen sie Millionen von Wanderern an, die auf der Suche nach unberührter Natur und atemberaubenden Panoramen sind. Im Winter verwandeln sich die Hänge in erstklassige Skigebiete. |
| Russian | Сегодня мы поговорим о великих открытиях в области космонавтики. В прошлом году учёные обнаружили новую экзопланету, которая находится в обитаемой зоне своей звезды. Это открытие может изменить наше понимание Вселенной. |
| Portuguese | O futebol é mais do que um esporte no Brasil — é uma paixão nacional. Desde as peladas nas ruas de areia até os grandes estádios, o jogo une milhões de pessoas em torno de um mesmo sonho: a vitória. |
| Arabic | أعلن فريق من العلماء اليوم عن اكتشاف مذهل في أعماق الصحراء الكبرى. عثر الباحثون على آثار مدينة قديمة يعود تاريخها إلى أكثر من خمسة آلاف عام، مما يغير فهمنا لتاريخ شمال أفريقيا. |
| Hindi | दीपावली भारत का सबसे रंगीन त्योहार है। हर घर दीयों की रोशनी से जगमगा उठता है, बच्चे पटाखे चलाते हैं, और हवा में मिठाइयों की मीठी खुशबू फैल जाती है। यह त्योहार अंधकार पर प्रकाश की जीत का प्रतीक है, और हर भारतीय के दिल में एक खास जगह रखता है। |
| Thai | ตลาดน้ำดำเนินสะดวกเป็นหนึ่งในสถานที่ท่องเที่ยวที่มีชีวิตชีวาที่สุดของไทย พ่อค้าแม่ค้าพายเรือขายก๋วยเตี๋ยว ผลไม้สด และขนมหวานตามสายน้ำ กลิ่นหอมของอาหารลอยมากับสายลม ทำให้ผู้มาเยือนรู้สึกเหมือนได้ย้อนกลับไปในอดีต |
| Indonesian | Bali bukan sekadar pulau — ia adalah dunia tersendiri. Sawah terasering di Ubud membentang seperti permadani hijau, sementara pura-pura kuno berdiri megah di tebing menghadap lautan. Setiap pagi, wangi dupa dan bunga persembahan memenuhi udara, mengingatkan kita bahwa di sini, tradisi dan alam hidup berdampingan dengan harmonis. |
| Hebrew | תל אביב היא עיר שלא נרדמת לעולם. מרחובות השוק הססגוניים של יפו ועד גורדי השחקים של רוטשילד, העיר פועמת באנרגיה ייחודית. בערב, חוף הים מתמלא באנשים שבאים ליהנות מהשקיעה, כשצלילי מוזיקה נישאים ברוח הים התיכון. |
| Vietnamese | Hà Nội vào thu đẹp lắm. Những hàng cây bàng lá đỏ rực rỡ hai bên đường, gió heo may nhẹ nhàng thổi qua mặt hồ Hoàn Kiếm. Người ta hay nói, muốn hiểu Hà Nội thì phải đến vào mùa thu. |

### Chinese Dialects

| Dialect | Target Text |
|:---|:---|
| 东北话 | 曾经有一份贼拉真挚的爱情搁在我跟前，我沒稀罕，等到整没了才后悔莫及。人世间最憋屈的事儿莫过于此 |
| 广东话 | 我从来都冇觉得我自己系介留守儿童，但系诶，根据我对留守儿童嘅了解，以及我目前嘅职业系一位乡村教师，我对留守儿童咧要更加自信嘅了解。我从细咧唔系跟住父母一齐住嘅，系同阿妈一齐生活嘅。喺嘅过程当中，父母巧少掌握计，我哋都巧少联系。喺读书 |
| 闽南语 | 透早起来，开门就听着厝角鸟仔咧啼，透早的空气，总是有彼股清芳的味。想着细汉的时阵，阿妈攑着葵扇，佇咧龙眼树下，共阮囡仔讲故事。彼种简单的快乐，如今想起来，心头阁是烧暖暖的。故乡的路，虽然已经阔甲袂记得形影，但是记忆中的香味，永远放袂记。 |
| 上海话 | 侬看看搿只小囡，机灵是机灵得来！一道算术题目，伊眼睛眨眨就解出来唻。勿像阿拉屋里个憨大，教十遍还囫囵转 |
| 河南话 | 今儿个天冷得邪乎！咱赶紧回去喝碗汤吧？ |
| 四川话 | 刘亚楼，你记一下，我做如下部署 |

---

## Expressive: Cross-Lingual Voice Transfer

Use the same reference audio and synthesize in different languages.

| Reference Language | Target Language | Target Text |
|:---:|:---:|:---|
| English | English | Hello everyone, welcome to the VoxCPM2 speech synthesis demo. Today I'll show you the effect of cross-lingual voice transfer in different languages. |
| English | Chinese | 大家好，欢迎来到VoxCPM2的语音合成演示。今天我将用不同的语言为你展示跨语言声音迁移的效果。 |
| English | Japanese | 皆さん、こんにちは。VoxCPM2の音声合成デモへようこそ。今日は異なる言語でクロスリンガル音声変換の効果をお見せします。 |
| English | French | Bonjour à tous, bienvenue dans la démonstration de synthèse vocale de VoxCPM2. Aujourd'hui, je vais vous montrer l'effet du transfert vocal multilingue. |
| English | Korean | 여러분 안녕하세요, VoxCPM2 음성 합성 데모에 오신 것을 환영합니다. 오늘 저는 다국어 음성 전환의 효과를 보여드리겠습니다. |
| English | Spanish | Hola a todos, bienvenidos a la demostración de síntesis de voz de VoxCPM2. Hoy les mostraré el efecto de la transferencia de voz multilingüe. |

---

## Creative: Voice Design from Natural-Language Description

No reference audio needed. Put the description in `voice_description` and the text in `text`.

| Voice Description | Target Text |
|:---|:---|
| Raspy old man | The world has changed, son. It ain't what it used to be. |
| Little girl, excited | 妈妈妈妈！你快来看！我画了一只大恐龙，它会喷火的！ |
| Pirate captain standing on the bow during a raging storm, shouting commands over the thunder and crashing waves | All hands on deck! Secure the mainsail! We ride this storm or we die trying — no turning back now, ye hear me? |
| Soft-spoken, breathy female voice with ASMR quality, extremely close-mic, every whisper feels like it's right next to your ear | Close your eyes and imagine you're lying on a warm beach. The waves are rolling in, one after another, gently washing over the sand. You're completely safe. Completely at peace. |
| Speaking through tears of joy | I can't believe it… we did it. After everything we've been through, we actually did it. I'm so proud of every single one of you. |
| 深夜电台主播，独自坐在昏暗的录音室里，窗外下着雨，用低沉温柔的嗓音陪伴听众度过失眠的夜晚 | 夜深了，如果你还醒着，不妨听我讲一个关于远方的故事。没有什么比雨声和一个好故事更适合这样的夜晚了。 |
| 女高音，音域明亮通透，共鸣集中在头腔，气息控制极稳，尾音带有轻微的颤音，声音干净无杂质，像清晨山间的溪水一样清澈 | 春江潮水连海平，海上明月共潮生。滟滟随波千万里，何处春江无月明。 |
| 强忍悲痛，声音在颤抖中努力保持平静，偶尔哽咽 | 我没事，真的。你不用担心我。只是……有些话我还没来得及跟他说。 |
| 一位相声演员，在舞台上进行表演，无背景噪声 | 今儿咱俩给大伙说段相声，相声讲究的是说学逗唱；那么什么是说呢？说就是说话，要求吐字清晰 |
| 中老年男声，苍老沙哑，音量不大，沉郁顿挫。气息悠扬，带着无奈和倦怠。 | 风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。万里悲秋常作客，百年多病独登台。艰难苦恨繁霜鬓，潦倒新停浊酒杯。 |
| 中年男声，是一位有文学素养的统军将领。音色浑厚、略微沙哑，胸腔共鸣很重。音质坚硬，有棱角，不是圆润的那种。声音中气十足，有穿透力，音强整体很大。尾音干脆利落。 | 怒发冲冠，凭栏处、潇潇雨歇。抬望眼，仰天长啸，壮怀激烈。三十功名尘与土，八千里路云和月。莫等闲、白了少年头，空悲切！靖康耻，犹未雪。臣子恨，何时灭！驾长车，踏破贺兰山缺。壮志饥餐胡虏肉，笑谈渴饮匈奴血。待从头、收拾旧山河，朝天阙！ |
| Cheerful young woman, warm smile in her voice, like a tour guide welcoming visitors on a sunny morning | Bonjour et bienvenue à bord ! Aujourd'hui, nous allons découvrir les plus beaux quartiers de Paris. Attachez vos ceintures et profitez du voyage ! |
| 温柔知性的中年女性，语速偏慢，像在读一封写给远方朋友的信 | Liebe Freunde, ich schreibe euch aus einer kleinen Stadt am Rhein. Der Herbst ist hier wunderschön — die Blätter leuchten in Gold und Rot, und der Fluss glitzert in der Nachmittagssonne. |
| Confident, energetic male sports commentator, fast-paced, building excitement with every word | ¡Recibe el balón en el medio campo, avanza con velocidad, dribla a uno, a dos — está solo frente al portero — ¡dispara! ¡GOOOL! ¡Qué jugada espectacular! |
| 沉稳大气的男性，像纪录片旁白，节奏从容，声音有磁性 | 何千年もの間、この土地は静かに歴史を見守ってきた。山々は変わらず、川は絶えず流れ、人々の暮らしだけが少しずつ姿を変えていった。 |
| 慈祥的老者，声音低沉浑厚，像在壁炉旁讲述古老的传说 | Много лет назад, когда леса были гуще, а ночи темнее, в этих горах жил старый мудрец. Говорят, он знал язык ветра и мог предсказать будущее по звёздам. |
| Song: Music, Piano, Sad, Female Vocal | 还没好好的感受 雪花绽放的气候 我们一起颤抖 会更明白 什么是温柔 还没跟你牵着手 走过荒芜的沙丘 可能从此以后 学会珍惜 天长和地久 |

---

## Controllable Voice Cloning: Same Voice, Different Styles

Use the same reference audio with different style instructions in `voice_description`.

| Style Instruction | Target Text |
|:---|:---|
| 轻声耳语 | 你应该也来试一下 VoxCPM 的。 |
| 四川话 | 你好像有好多问题想问，但是莫急，我们嘞时间多得很。 |
| Angry tone, volume gradually increased | 今天，就是你我决一死战的时刻，我要在这里让你痛不欲生。 |
| 语气中带着明显的愤怒，音量逐渐加大，想要用气势压倒对方。 | 今天，就是你我决一死战的时刻，我要在这里让你痛不欲生。 |
| Cheerful and laughing, as if sharing good news with a close friend | I just got the best news — you won't believe what happened today! Everything worked out perfectly, even better than we planned. |
| Song: Pop Music, Beat, Happy, Passion | I wanna make America great again, Awaken the faith that we forever retain. No matter the trials, the storms, or the rain, We'll cross every mountain and conquer the plain. I wanna make America great again! Oh! |
