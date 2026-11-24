# 演算法調查：開源求解器與文獻對照

日期：2026-11-24。範圍：我們的離線 CUDA 工具（`nvcc/`）與現有開源 poker/CFR 程式碼、論文的比較。

本文每一條關於程式碼的陳述都附檔案與行號（外部 repo 在 `/home/posetmage/download/github/poker/.tmp/<repo>/`，只讀不執行；OpenSpiel 與 slumbot2019 另外淺層 clone 到 worktree 的 `.tmp-survey/`，同樣只讀、未建置、未提交）。文獻條目附 URL，並標明「讀過」或「只看摘要」。README 上的行銷說法（例如「比 Pio 快」）一律標為未驗證。

## 0. 結論先講

1. **求解器本體（CFR+）不是我們的問題。** 30,000 次迭代後 exploitability 已低於 1e-5 bb（`nvcc/README.md`「Measured」），而且和 BSD-2 的 poker-cfr 在相同樹、相同（原始）權益下一致到 4 位小數。換成 Discounted CFR 對這棵小樹沒有實質好處；樹變大（多尺寸、6 人桌）時才值得。
2. **最大的不確定性在收益模型，不在演算法。** 「call 之後」的價值用一個固定公式 `realised_ip`（`nvcc/solve.cuh:31-32, 234-235`）。我們已經做了實現係數表（`nvcc/realization.cuh`），但求解器還沒讀它（`PLAN.md` 第 378 行附近的限制），而且那張表本身是固定打法模擬，沒有和任何真正的翻牌後求解比對。已調查的開源程式裡，沒有一個提供可直接比對的實現係數（見 3.3）。
3. **6 人桌：沒有任何理論保證。** Pluribus 補充材料明說 CFR 的平均策略在 6 人局不保證收斂到 Nash（見第 3 節）。我們的 `exploitability` 在多人下沒有意義，需要換成「1 對 n-1 的最佳反應」類指標（slumbot2019 的做法）。
4. **有一個和已發表實驗設定不同的地方：** 我們的 CFR+ 是「同步更新」，DCFR 論文與所有被讀過的求解器都是「交替更新」。對現在的樹無影響（已收斂），但這是已知差異，見 3.1。

## (a) 來源表

「讀」＝實際讀過程式碼／全文；「略讀」＝只看摘要或搜尋結果摘要；「未讀」＝只看到名字。

| 來源 | URL | 授權 | 實際讀了什麼 | 我們能學什麼 | 授權限制 |
| --- | --- | --- | --- | --- | --- |
| b-inary/poker-cfr（Rust） | https://github.com/b-inary/poker-cfr | BSD-2（`LICENSE.md`） | 讀：`src/cfr.rs`、`src/game_preflop.rs`、`src/main_preflop.rs` | 單挑翻牌前 CFR+、1326 組合層級、全部 call 都用原始權益（check 到攤牌）、乘數式加注階梯。我們已用它做過外部對照 | 可細讀、可借寫法 |
| b-inary/postflop-solver（Rust） | https://github.com/b-inary/postflop-solver | AGPL-3.0（`LICENSE` 開頭） | 讀：`src/solver.rs`、`src/utility.rs`（exploitability、同構、壓縮）、`src/game/evaluation.rs`、`src/action_tree.rs`（部分）；`bunching.rs` 只確認存在，未讀 | DCFR 的具體參數與策略重置、16-bit 壓縮、花色同構、下注尺寸合併、全下門檻 | 只能借想法，不可複製程式碼 |
| TexasSolver（C++） | https://github.com/bupticybee/TexasSolver | AGPL-3.0（`LICENSE` 開頭） | 讀：`src/solver/PCfrSolver.cpp`、`src/trainable/DiscountedCfrTrainable.{h,cpp}`（部分） | DCFR 的另一組參數（含 theta 衰減）、warmup 用蒙地卡羅、omp 多執行緒、以 accuracy 停止 | 只能借想法 |
| GPU-CFR（Li & Huang，論文 arXiv:2609.11923） | README 內 badge 指向 CC BY-NC-SA 4.0 | CC BY-NC-SA 4.0（僅 README 宣稱；目錄內沒有 `LICENSE` 檔） | 讀：`README.md`、`BUILD_INFO.txt`、`gpu_cfr/kernels/triton_fused.py` 開頭。**核心求解器是編譯過的 `.so`（69 個），原始碼不可讀**；`probes/*.py` 只是載入編譯模組的 stub；zip 檔未解開 | 只能確認：它支援 CFR/CFR+/DCFR/PCFR+、同步與交替更新兩種（README「Running the benchmarks」`--alternating-updates`）、用 CUDA graph 重播（和我們 `solve.cuh:536-543` 同一招）。「80 倍」是標題宣稱，未驗證，也未讀論文全文 | 非商用；不要用它的程式碼。論文內容未讀，不能當依據 |
| NVIDIA/cuda-rust | https://github.com/NVIDIA/cuda-rust | Apache-2.0（`LICENSE`） | 讀：README 開頭；`grep -w cfr\|poker` 在 `*.rs/*.md` 無任何命中 | 與撲克無關（Rust 寫 CUDA kernel 的工具鏈）。我們用 CUDA C++，沒有理由採用 | 無 |
| OpenSpiel（Google DeepMind） | https://github.com/google-deepmind/open_spiel | Apache-2.0（`LICENSE`） | 讀：`python/algorithms/cfr.py`、`discounted_cfr.py`、`exploitability.py`；C++ 端只看檔名 | 參考實作：CFR+ ＝ regret matching+ ＋ 交替更新 ＋ 線性平均（`cfr.py:470-510`）；DCFR 預設 α=3/2, β=0, γ=2（`discounted_cfr.py:212-222`）；exploitability 只接受兩人零和（`exploitability.py:139-149`） | 可細讀 |
| slumbot2019（Eric Jackson，C++） | https://github.com/ericgjackson/slumbot2019 | MIT（`LICENSE`） | 讀：`README.md`（Features、Multiplayer、Discussion）、`notes/multiplayer`、`notes/mp3_results`、`notes/mp6_results`、`notes/abstraction`；`src/` 只看檔名 | 作者自己的實驗筆記：3 人與 6 人小遊戲的結果、「1 對 n-1 最佳反應」的想法；README 說明多人只支援 MCCFR 不支援 CFR+、不能算真實遊戲最佳反應 | 可細讀 |
| Brown & Sandholm, DCFR（AAAI 2019） | https://arxiv.org/abs/1809.04040（PDF 全文抽字） | — | 讀：推薦參數、交替更新、CFR+ 用 t² 平均 | 見 3.1、3.2 | — |
| Brown & Sandholm, Pluribus Science 補充材料 | https://noambrown.com/papers/19-Science-Superhuman_Supp.pdf | — | 讀：行動抽象、資訊抽象、Linear MCCFR、藍圖平均、多人不保證收斂（我用自寫的簡易 PDF 抽字，空白和符號有損，數字已對照上下文） | 見 3.4、3.5、3.6 | — |
| Tammelin et al., CFR+ | https://arxiv.org/abs/1407.5042 | — | 略讀：只看摘要，摘要沒有講平均方式與更新順序 | 「比舊演算法快一個數量級」是作者宣稱，未驗證 | — |
| Hyperparameter Schedules（DCFR 後續） | https://arxiv.org/html/2404.09097v2 | — | 略讀（網頁摘要） | 作者宣稱在 HUNL 殘局等 10 個遊戲上，γ 隨迭代遞減的排程勝過 DCFR；實驗用交替更新。數字（「3.1 個數量級」）來自摘要，未驗證 | — |
| Risk & Szafron；Gibson（多人 CFR） | https://www.cs.ualberta.ca/~games/poker/publications/AAMAS10.pdf ；https://ar5iv.labs.arxiv.org/html/1305.0034 ；https://arxiv.org/pdf/1804.04789 | — | 略讀（搜尋結果摘要） | 多人 CFR 無理論保證，但實務上 3 人 Hold'em 代理仍可贏；3 人 Kuhn 收斂、3 人 Leduc 未必 | — |
| Poker Academy EQR 部落格 | https://poker.academy/blog/post/how-to-understand-equity-realisation-eqr | 商業部落格 | 略讀（搜尋結果摘要） | 定義：實現率 ＝ 實拿底池比例 ÷ 攤牌權益；舉例 BTN 對 BB、20bb：IP 約 1.15、OOP 約 0.80。廠商文章，未同行審查，**不當事實** | — |
| Pokernews 對 HRC 與 Simple Preflop Holdem 的比較 | https://www.pokernews.com/strategy/bb-defense-holdem-resources-vs-simple-preflop-holdem-38183.htm | 新聞文章 | 略讀（搜尋結果摘要） | 稱 Holdem Resources Calculator 假設最後一次加注後直接 check 到攤牌；Simple Preflop Holdem 用位置與翻牌後下注變數估實現。**二手說法，未驗證**；Monker 和 GTO Wizard 的內部做法沒找到可靠來源 | — |

「cuda-rust」和「GPU-CFR」是這次任務點名的，兩者對我們的直接價值都很低，原因如上，不是沒讀。

## (b) 我們的選擇 vs 通行做法

（以下小節編號 3.1 到 3.8 是本節內部編號，文中其他地方的「見 3.x」指這裡。）

### 3.1 CFR+ vs Discounted CFR

| 項目 | 我們 | 其他實作 | 判斷 |
| --- | --- | --- | --- |
| 後悔值 | 加總後截到 0（`solve.cuh:376`） | poker-cfr 同（`cfr.rs:154-158`）；OpenSpiel CFR+ 同（`cfr.py:470-510`） | 一致 |
| 平均權重 | 迭代次數 t（`solve.cuh:377`） | DCFR 論文：CFR+ 用 t² 比 t 快（實驗觀察，PDF 全文：「Empirically we observed that CFR+ converges faster when assigning iteration t a weight of t²」）。poker-cfr 用 t（`cfr.rs:163`，從 0 起算，第一次權重為 0） | 我們用 t，和 CFR+ 原版、OpenSpiel 一致；t² 是 DCFR 論文的經驗改進，**沒測過在我們的樹上有沒有差** |
| DCFR | 無 | postflop-solver：α_t=t^1.5/(t^1.5+1)、β=0.5、γ 項為 ((t−t₄)/(t−t₄+1))³，t₄ 是不大於 t 的最大 4 的冪，也就是**每到 4 的冪就重置平均**（`solver.rs:11-36`；README 實作細節也寫了 γ=3、4 的冪重置）。TexasSolver：α=1.5、β=0.5、γ=2、另有 θ=0.9 的累計策略衰減（`DiscountedCfrTrainable.h:20-23`、`.cpp:89-130`）。OpenSpiel 預設 α=1.5、β=0、γ=2（`discounted_cfr.py:212-222`） | 三個程式用三組不同參數，說明沒有單一公認最佳。論文實驗：DCFR(3/2,0,2) 對 CFR+「matches or outperforms across the board」、通常快 2 到 3 倍；DCFR(3/2,1/2,2) 在 Goofspiel 明顯較差（取自 ar5iv 版摘要，非逐字核對） |
| 更新順序 | **同步**：一次 top-down、一次 bottom-up，兩位玩家同時更新（`solve.cuh:527-531`） | poker-cfr：交替（`cfr.rs:504-505`）；postflop-solver：交替（`solver.rs:75-92`）；TexasSolver：交替（`PCfrSolver.cpp:~795-806` 的 `for player_id`）；OpenSpiel：`alternating_updates=True`（`cfr.py:509-510`）；DCFR 論文：「in practice far better performance is achieved by alternating」，所有實驗用交替 | **我們是唯一同步的**。GPU 版選同步是因為一次 sweep 同時算兩邊值較省，但已發表的 CFR+/DCFR 速度數據都是交替的。GPU-CFR 兩種都支援（README），但沒有可讀原始碼說明差距多大。對我們現狀無實害（已收斂），樹變大時要測 |
| 平均策略 | 乘上自己的到達機率（`solve.cuh:377`，標準寫法） | postflop-solver 累加的是當下策略本身，沒有乘自己的到達機率（`solver.rs:302-306`，`*x = *x*gamma + *y`，`y` 是 `strategy`）；TexasSolver 同樣不乘（`DiscountedCfrTrainable.cpp:122-128` 把 `* reach_probs` 註解掉） | 兩個專業翻牌後求解器都不乘。我**沒有驗證**兩種寫法的差異（尤其在很少到達的節點）。對我們：低到達的手牌 `k_average` 退回均勻（`solve.cuh:389`），輸出已用 reach<0.01 過濾（`solve.cuh:677-678`） |
| 採樣 | 全樹窮舉（向量化） | slumbot2019 建議能窮舉時用 CFR+（README「Discussion」）；Pluribus 用 Linear MCCFR（補充材料） | 一致 |

### 3.2 收斂判準

- 我們：從兩個座位算最佳反應值，單位 bb（`solve.cuh:567-580`）。它量的是「對這個模型的可剝削度」：模型是 169 類別、固定實現公式。輸出 JSON 的欄位四捨五入到 3 位，所以顯示 0（`nvcc/README.md`「Use」）。
- postflop-solver：兩邊最佳反應值的平均（`utility.rs:279-291`）；每 10 次迭代算一次，達到目標就停（`solver.rs:67-87`）。TexasSolver：每 `print_interval` 次算一次，`<= accuracy` 就停（`PCfrSolver.cpp:802-816`）。兩者的單位（絕對籌碼或相對底池）我沒有確認，**未驗證**。
- OpenSpiel：只支援兩人零和（`exploitability.py:139-149`）。
- 對我們的含意：停止條件目前是固定迭代數（30,000），而實測在 1e-5 bb 以下，過度；可改成「達標即停」省時間，但好處很小。真正缺的是**不依賴單一收斂數字的驗證**：不同種子、不同初始化重跑，邊緣手牌的頻率是否一致。

### 3.3 實現係數

| 項目 | 我們 | 其他做法 | 判斷 |
| --- | --- | --- | --- |
| 求解器內使用 | 單一公式 `0.5 + 1.08·(eq−0.5) + 0.025`，只有 SB（IP）一方；BB 取 1−該值（`solve.cuh:31-32, 234-235`），對所有手牌、所有底池類型、所有深度相同；全下的 call 用原始權益（`solve.cuh:326`） | poker-cfr：全部 call 當成直接攤牌，用原始權益（`game_preflop.rs:114-134`）。HRC 據稱同（略讀，二手）。據稱 Simple Preflop Holdem 依位置與下注變數（略讀，二手） | 我們比 poker-cfr 多了一個粗糙的修正。這個修正沒有任何外部校準，而且**對所有手牌同向同幅**（slope 1.08），把弱牌（eq<0.5）也推得更低、強牌更高；實際上同花連張這類 eq 中等但實現好的牌會被低估 |
| 離線表 | `realization.cuh`：固定政策模擬（價值下注／詐唬比例、MDF 防守、一種尺寸）、依類別×位置×底池類型×深度，隨機牌面 20 萬個（`PLAN.md` 完成狀況） | 沒有找到開源程式提供對照 | 這是我們的獨有部分；已知限制寫在 `PLAN.md`「限制（必讀）」與 `nvcc/README.md:475` |
| 校準 | 無 | 唯一可行的外部真值是真正的翻牌後求解（postflop-solver / TexasSolver，皆 AGPL）或自己寫 | 見 (c) 第 2 項 |

### 3.4 多人（6 人桌）

- Pluribus 補充材料（「Further details of the abstraction algorithm」之後的訓練段落）：「Since CFR's average strategy is not guaranteed to converge to a Nash equilibrium in six-player poker, there is no theoretical benefit to using the average strategy as opposed to the current strategy.」他們翻牌前（第一輪）仍存平均策略當藍圖，之後每輪存 200 分鐘一次的當前策略快照再平均。
- slumbot2019：README 說多人只支援 MCCFR、不支援 CFR+，也不能算真實遊戲最佳反應（README「Multiplayer」）。作者在 `notes/multiplayer` 構想「1 對 n-1 最佳反應」：固定其他 n−1 位的策略，對剩下一位算最佳反應，每個座位各做一次。`notes/mp3_results` 顯示他在一個小的 3 人遊戲上用這個方法量到每座位 3.2、7.6、3.5 mbb/g，總 exploitability 4.74 mbb/g（作者自己的筆記，我沒有重跑，也沒讀 `run_mp_rgbr` 的程式碼）。
- 文獻（只讀摘要）：3 人 Kuhn CFR 收斂，3 人 Leduc 不保證；Gibson 的 3 人 Hold'em 代理仍有實戰成績。
- 我們的現況：6 人桌求解尚未做，正由其他代理規劃「迭代最佳反應」。這和 Pluribus／slumbot 的做法精神一致（沒有保證，靠實測指標），但**必須明講沒有收斂保證**，並採用 1 對 n−1 的偏離收益作為指標，而不是 `exploitability`（兩人零和才有意義）。

### 3.5 下注尺寸集合與樹

| 項目 | 我們 | 其他 |
| --- | --- | --- |
| 翻牌前階梯 | 開池 `[2.5]`、3bet `[9]`、4bet `[22]` 各一種；超過 70% 籌碼視為全下（`solve.cuh:62-100`，門檻在第 84 行） | poker-cfr：對方下注的 2.5/3/3.5/4 倍加全下，並用「剩餘籌碼比」限制可用倍數（`game_preflop.rs:40-74`）。Pluribus：藍圖每個決策點 1 到 14 種加注尺寸，**第一輪最細**，因為第一輪不做即時搜尋；尺寸由手工挑選，依據是更早版本的 Pluribus 常用哪些尺寸（補充材料） |
| 全下處理 | 70% 規則（`solve.cuh:84`） | postflop-solver：`add_allin_threshold`（下注超過底池比例的門檻就加入全下）、`force_allin_threshold`（`action_tree.rs:117, 586, 618, 658`）；`merging_threshold` 合併過近的尺寸（`action_tree.rs:133, 700`） |
| 離樹動作 | 沒有（求解結果只在樹內有效） | Pluribus 搜尋時把對手的離樹動作加入子博弈（補充材料）。我們的網站使用情境是「使用者查表」，離樹動作需要查表時映射到最近節點，目前沒有規格 |

### 3.6 深度與抽象

- 我們：每個深度獨立一棵樹，同時打包在 GPU 上（`solve.cuh:12-16`）。這等於「每個有效籌碼一個獨立博弈」，是合理的。
- 抽象：手牌 169 類是無損抽象（Pluribus 補充材料：第一輪 169 種策略上不同的資訊情境，且只在第一輪用無損抽象）。我們的「無損」只對**原始權益矩陣**成立；對策略而言類別內的花色組合因為卡牌移除（W 矩陣）不完全對稱。我們已自己驗證：和 poker-cfr（1326 組合）在相同樹、原始權益下，SB 的價值吻合到 4 位小數、機率加權的平均 TV 距離 0.0005 到 0.0045（`nvcc/README.md`「test_solve_external」表）。所以**類別層級對單挑原始權益影響很小**；對實現係數與三人權益的影響沒有量過。
- postflop-solver 不做任何抽象，只合併同構的發牌（`README.md`「Isomorphism」、`utility.rs:420-426`）。

### 3.7 數值精度與壓縮

- 我們全程 double、`-fmad=false` 以便和 TypeScript 逐位元一致（`solve.cuh:17-19`）。對 169 類別小樹沒有壓力。
- postflop-solver：多數用 f32、求和用 f64；可選 16 位元整數壓縮加每節點一個 f32 縮放（`solver.rs:250-283`、`utility.rs:204-228`；README「Precision」）。對我們無用，除非做翻牌後求解（記憶體壓力才出現）。

### 3.8 呈現混合頻率

- 我們輸出每手牌類別在每節點的頻率，四捨五入到 2 位，reach<0.01 的類別省略；節點到達機率<0.0005 省略（`solve.cuh:668-678`）。
- 未到達的手牌：`k_average` 對從未到達的手牌輸出均勻 1/k（`solve.cuh:389`）。這不是求出來的策略，網站若直接顯示會誤導；需要在 JSON 或前端標示「此手牌在此節點機率為 0」。
- Pluribus 的做法提醒：混合頻率只在到達機率夠高時有意義。

## (c) 具體改進，依效益排序

預期效益只是我的推估，除非另有標註。

| # | 改動 | 預期效益 | 成本 | 會動到的檔案 |
| --- | --- | --- | --- | --- |
| 1 | **求解器讀實現係數表**，取代 `realised_ip`；同時讓 BB（OOP）自己有係數而不是 `1−` SB 的值；兩邊加總不等於 1 時需要處理（見第 (d) 節第 3 點）。`PLAN.md` 已列為下一步第 1 項 | 最大：目前所有 call 價值都來自一條沒校準的直線。邊緣手牌（疑似同花連張、A-x）的頻率會明顯變動 | 中：改 `make_problem`，把 `M[1]` 依底池類型（單加注／3bet／4bet）與深度分別建矩陣；樹節點要帶「底池類型」欄位 | `nvcc/solve.cuh`（`make_problem`、`Problem`、`DevNode`、`k_show`）、`nvcc/solve.cu`、`tools/solve-preflop.ts`（TS 參考實作要同步，否則 `test_solve` 的逐位元對照會失效）、`nvcc/test_solve.cu` |
| 2 | **用真正的翻牌後求解校準實現係數**：挑幾個代表性翻牌與範圍（用我們自己 100bb 單挑解的範圍），拿 postflop-solver 或 TexasSolver 當**外部程式**離線跑，比對 EQR（實拿底池比例÷權益）與我們表中的對應數字。只取數字，不貼程式碼；兩者皆 AGPL，執行別人的 AGPL 程式並使用其輸出通常不構成衍生，但**這是法律判斷，我沒有核實，使用前請確認**。另一條路：用 `solve.cuh` 的向量化結構自己寫 GPU 翻牌後 CFR | 這是目前唯一能回答「實現係數對不對」的辦法；也能找出系統性偏差（OOP 低估、深籌碼強牌低估，已知於 `PLAN.md` 限制） | 高：翻牌後單一翻牌求解要幾 GB 與數分鐘（`PLAN.md`GPU 章節的估計）；比對 20 個翻牌以上才有統計意義。**注意：該工具要自己確認能在此機器上跑，我沒有跑它** | 新增 `nvcc/` 下的比對工具或腳本；`nvcc/realization.cuh`（若要調政策參數 `default_pol`） |
| 3 | **閉環：用求解出的範圍重算實現係數**，再重解，迭代到穩定（固定點）。目前同一組 100bb 範圍用在所有深度，20bb 偏寬（`PLAN.md` 限制） | 中到高：範圍相依的係數是最明顯的內在不一致 | 中：`realization.cu` 要能接受範圍檔輸入（它已有範圍權重輸入 `Blk::w0`）；跑 3 到 5 輪，每輪 7 分鐘級（每種底池類型） | `nvcc/realization.cu`、`nvcc/realization.cuh`、`nvcc/solve.cu`（輸出範圍檔）、`data/realization.json` 格式 |
| 4 | **6 人桌的驗證指標：1 對 n−1 偏離收益**（仿 slumbot2019 `notes/multiplayer`）。固定其他 5 個座位的平均策略，對剩下一個座位算最佳反應，對每個座位各一次；回報最大值與總和。另外記錄「當前策略」與「平均策略」兩者各自的指標（Pluribus 的結論是多人下兩者無理論差別） | 對 6 人桌是必要的：沒有它就無法說「這個解有多可信」，而 `exploitability` 在多人下不成立 | 中：需要多人樹的最佳反應 kernel；`solve.cuh` 的 `mode 2` 可擴展 | 新工具檔（例如 `nvcc/solve6.cuh`）、對應 `test_*.cu`；不影響現有單挑檔 |
| 5 | **改交替更新（並可選 DCFR）**，先在現有單挑樹上量收斂速度（迭代數到 1e-4 bb） | 對現在的樹約為零（已達 1e-5 bb）；對多尺寸、6 人桌的大樹可能有 2 到 3 倍（DCFR 論文 AAAI 摘要：「factor of 2 or 3」，**是它們遊戲上的數字，不是我們的**） | 低到中：交替更新要把一次迭代拆成兩次 sweep（每個玩家一次），CUDA graph 要重錄；DCFR 只多幾個乘法，但 TS 參考實作與 `test_solve` 的「逐位元一致」要放棄或另立測試 | `nvcc/solve.cuh`（`enqueue_iteration`、`k_bottomup`）、`nvcc/test_solve.cu`、`tools/solve-preflop.ts` |
| 6 | **多尺寸階梯與自動刪減**：每層 2 到 3 種尺寸，跑完後刪掉使用率低的（Pluribus 的作法是依先前版本的使用頻率挑，補充材料）；全下門檻改成 `add_allin_threshold` 式（相對底池）而非固定 70% 籌碼 | 中：網站使用情境需要多種尺寸，但每增加一種樹會倍增 | 低：`build_ladder_tree` 已支援 `--open 2/2.5,9` 的備選語法（`nvcc/README.md`「Use」） | `nvcc/solve.cu` 參數、`nvcc/solve.cuh:62-100` |
| 7 | **把「沒到達」從輸出明確標出**，不要輸出 1/k 的均勻值當策略 | 小但沒有風險；避免網站畫出假混合頻率 | 低 | `nvcc/solve.cuh`（`json`，第 654-696 行）、`src/` 內讀取與顯示的程式 |
| 8 | **不同種子重跑與穩定性報告**（隨機初始化的後悔值，或打亂迭代順序） | 驗證邊緣手牌頻率是否是求解雜訊；`PLAN.md` 優先順序第 4 項已有此想法 | 低：39 秒跑 11 個深度 | 新增 `nvcc/test_*` 或 `solve.cu` 選項 |
| 9 | 把平均權重試成 t²（DCFR 論文對 CFR+ 的建議）並比較 | 可能小幅加速；樹已收斂所以不急 | 低 | `nvcc/solve.cuh:377` 與 TS 參考實作 |

不建議做的事：
- **不要為了「GPU-CFR 號稱 80 倍」改架構。** 我們已用 CUDA graph 重播，單挑 11 個深度 30,000 次迭代 39 秒；瓶頸不在求解器。
- **不要直接改成 1326 組合層級的單挑求解。** 我們自己的外部比對顯示類別層級差距極小。
- **cuda-rust**：與撲克無關，沒有採用理由。

## (d) 已知陷阱

1. **exploitability 只對模型成立。** 它量的是對「169 類別 + 固定實現公式」這個博弈的可剝削度，不是對真實撲克（`PLAN.md` 已說，外加雜訊矩陣的警告）。顯示 0 不是「解對」。
2. **類別層級的卡牌移除。** 單挑原始權益用衝突組合數 W 處理（`solve.cuh:217-226`），誤差很小（見 3.6）。三人權益表對「同類別各花色權重相同」的範圍精確，其他情況有誤差（`equity3.cuh` 檔頭、`PLAN.md`）。多人桌還有**已棄牌玩家的牌**造成的移除（postflop-solver 專門做了 bunching 效應，README「Bunching effect」、`src/bunching.rs`，我沒讀實作）；我們的表假設其他玩家的牌不影響，對 6 人桌是系統性偏差，尤其會影響 SB／BB 的範圍價值。
3. **實現係數是否零和。** `solve.cuh:234-235` 用 IP 的實現公式，OOP 取 `1−`，所以零和成立。換成表之後，IP 與 OOP 各自的實現係數**不再保證**在同一手牌對上加總為 1，CFR+ 在非零和的收益上沒有保證。要嘛強制歸一，要嘛承認偏離並重新驗證。`realization.cuh` 檔頭說「chips are exact per deal: on every deal the two sides' chips won add up to the final pot」，意思是在**每一副牌**上加總成立，但按類別取平均後比值（real/raw）不見得一致，這點我沒有核實。
4. **範圍外手牌的係數偏高**（最高 1.7，`nvcc/README.md:475`）。求解時範圍會變，若某手牌原本不在模擬範圍但解中開始玩，係數會系統性偏高，形成虛假的吸引力。閉環（第 (c) 第 3 項）是解法。
5. **平手處理。** 三人權益表：平手平分，三人平手各 1/3（`equity3.cuh:5`）。兩人權益沿用同一原則。postflop-solver 明確處理平手（`evaluation.rs:154` 附近的 `amount_tie`）。我們未量過「平手權重」對類別層級的影響，預期很小。
6. **3 人權益沒有 ICM。** 我們是現金局（`PLAN.md` 第 8 項說明）；若未來做錦標賽，3 人以上的籌碼價值不能用權益 × 底池。
7. **三人權益表與實現係數的接軌缺口。** 三人權益是「全下攤牌」的權益；實現係數只做單挑。多人底池翻牌後的實現沒有模型：用單挑係數乘三人權益會高估 OOP 的牌。目前沒有任何開源程式碼可對照，這是我們必須自己面對的缺口。
8. **多人下的平均策略。** Pluribus 補充材料明確指出平均策略沒有理論優勢；用平均還是用當前策略要實測，不要預設。
9. **GPU-CFR 授權與可讀性。** README 的 CC BY-NC-SA 標示沒有對應 LICENSE 檔，且主要程式是編譯產物。不要使用，也不要依賴它的論文數字。
10. **README 行銷用語。** postflop-solver README 稱「surpasses paid solvers such as PioSOLVER and GTO+」（`README.md` 實作細節段），TexasSolver README 稱「speed exceeds piosolver on flop」。兩者我都沒有驗證，本文不引用為事實。

## (e) 沒能釐清的問題

1. **Monker、GTO Wizard、HRC、Simple Preflop Holdem 的實際翻牌後近似方式**：只找到二手說法，沒有一手資料。所以「商用翻牌前求解器怎麼估實現」沒有被驗證；本文第 3.3 節對這部分的陳述都標為二手。
2. **有沒有公開論文專門估「依類別、位置、SPR 的實現係數」**：我搜尋兩次沒找到，不代表不存在。
3. **平均策略是否要乘自己的到達機率**：postflop-solver 與 TexasSolver 都沒乘，標準 CFR 乘。我沒有證明兩者的收斂是否等價（直覺上差別在低到達機率的資訊集）。
4. **DCFR 的 postflop-solver 版本（γ=3、4 的冪重置）為何這樣選**：README 只說明了選擇，沒有說依據；我沒有找到其基準。
5. **GPU-CFR 論文是否有可轉用於「兩人小樹 169 類別」的結果**：沒讀論文，核心程式碼不可讀。
6. **交替更新對我們的 GPU 版本多慢／多快**：需要實驗，沒有跑。
7. **postflop-solver 與 TexasSolver 的 exploitability 單位**（絕對或相對底池）：沒確認。
8. **以 postflop-solver／TexasSolver 當外部程式產生校準數據的授權解讀**：見 (c) 第 2 項，需要有人確認。
9. **Pluribus 補充材料 PDF 文字是自寫的簡易抽字，可能有漏字或符號錯亂**：引用的數字（200 buckets、400 分鐘、1 到 14 種尺寸）已和上下文核對，但非逐頁人工對照。
