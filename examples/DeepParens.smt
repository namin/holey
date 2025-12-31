(set-logic ALL)

(declare-datatypes ((List 1))
    ((par (T) ((cons (head T) (tail (List T))) (nil)))))


(define-fun-rec list.length.int ((l (List Int))) Int
  (ite (= l (as nil (List Int)))
       0
       (+ 1 (list.length.int (tail l)))))


(define-fun-rec list.get.int ((l (List Int)) (idx Int)) Int
  (ite (< idx 0)
       (list.get.int l (+ (list.length.int l) idx))
  (ite (= idx 0)
       (head l)
       (list.get.int (tail l) (- idx 1)))))

(declare-const x (List Int))
(assert (> (list.length.int x) 0))
(assert (>= (- (list.get.int x 0) 1) 0))
(assert true)
(assert (>= (- (+ (- (list.get.int x 0) 1) 1) 1) 0))
(assert true)
(assert (>= (- (+ (- (+ (- (list.get.int x 0) 1) 1) 1) 1) 1) 0))
(assert (>= (- (- (+ (- (+ (- (list.get.int x 0) 1) 1) 1) 1) 1) 1) 0))
(assert true)
(assert true)
(assert (>= (- (+ (+ (- (- (+ (- (+ (- (list.get.int x 0) 1) 1) 1) 1) 1) 1) 1) 1) 1) 0))
(assert true)
(assert (>= (- (+ (- (+ (+ (- (- (+ (- (+ (- (list.get.int x 0) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 0))
(assert (>= (- (- (+ (- (+ (+ (- (- (+ (- (+ (- (list.get.int x 0) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 0))
(assert true)
(assert true)
(assert (ite (= (- (- (+ (- (+ (+ (- (- (+ (- (+ (- (list.get.int x 0) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 0) true (ite (= (- (+ (- (+ (+ (- (- (+ (- (+ (- (list.get.int x 0) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 0) true (ite (= (- (+ (+ (- (- (+ (- (+ (- (list.get.int x 0) 1) 1) 1) 1) 1) 1) 1) 1) 1) 0) true (ite (= (- (- (+ (- (+ (- (list.get.int x 0) 1) 1) 1) 1) 1) 1) 0) true (ite (= (- (+ (- (+ (- (list.get.int x 0) 1) 1) 1) 1) 1) 0) true (ite (= (- (+ (- (list.get.int x 0) 1) 1) 1) 0) true (ite (= (- (list.get.int x 0) 1) 0) true false))))))))
(assert (= (list.length.int x) 1))
(check-sat)
(get-model)