(set-logic ALL)

(define-fun-rec str.count.rec ((s String) (sub String) (start Int)) Int
  (let ((idx (str.indexof s sub start)))
    (ite (or (= idx (- 1)) (> start (str.len s)))
         0
         (+ 1 (str.count.rec s sub (+ idx (str.len sub)))))))

(define-fun str.count ((s String) (sub String)) Int
  (ite (= (str.len sub) 0)
       (+ 1 (str.len s))
       (str.count.rec s sub 0)))

(declare-const x String)
(assert (>= (str.count x "o") 0))
(assert (>= (str.count x "oo") 0))
(assert (and (= (str.count x "o") 1000) (= (str.count x "oo") 0)))
(check-sat)
(get-model)
