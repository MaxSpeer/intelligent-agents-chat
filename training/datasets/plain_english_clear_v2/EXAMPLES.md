# Beispiele aus den neuen Trainingsdaten

Diese Antworten sind tatsächlich in `train.jsonl` enthalten. Es sind neue Musterantworten, keine Ausgaben eines neu trainierten Modells. Die Beispiele zeigen bewusst verschiedene Themen.

## plain-en-0081

**Frage:** What is a real-world situations that can be modeled with a two variable linear equation? What would the y-intercept, x-intercept, and slope of the line represent?

**Vorher:**

Suppose you earn $10 per hour. Let x be the hours worked and y be the money earned. The relationship is y = 10x.

The slope is 10: each extra hour adds $10.

The y-intercept is 0: when x is 0, you have earned $0. The x-intercept is also 0: earnings are $0 when you have worked 0 hours.

Both intercepts are at the origin, the point (0, 0). This model assumes a constant hourly rate with no fixed payment.

**Clear v2:**

Suppose you earn $10 for each hour of work. Let x mean hours worked. Let y mean money earned. The equation is y = 10x.

The slope tells you how fast the money increases. Here it is 10: each extra hour adds $10.

The y-intercept is where the line crosses the money axis. Here it is 0. Before working any hours, you have earned $0.

The x-intercept is where the line crosses the hours axis. It shows the hours worked when earnings are $0. Here that is also 0. Both crossings are at (0, 0), called the origin. This example assumes a fixed hourly rate and no separate fixed payment.

## plain-en-0065

**Frage:** what is batik?

**Vorher:**

Batik is a way to decorate fabric using wax and dye. Wax is applied to parts of the cloth, and those areas resist the dye. When the wax is removed, a pattern appears because the protected areas keep their earlier color.

The process can be repeated with more wax and different dyes to build up a design. Batik is especially associated with Indonesia, where patterns can carry cultural meanings, though the technique is also practiced elsewhere.

**Clear v2:**

Batik is a way to decorate cloth using wax and dye. Dye is a substance that adds color to the cloth. First, put wax on parts of the cloth. These parts resist the dye. Remove the wax to reveal a pattern. The covered parts keep their earlier color.

You can repeat these steps with more wax and different dyes. Batik is especially linked to Indonesia. Its patterns can carry cultural meanings there. People also practice it in other places.

## plain2k-everyday-train_sft-2017-qa2

**Frage:** I'm learning about simple machines in school. What's a compound machine?

**Vorher:**

A compound machine is a machine that combines two or more simple machines to work together. Examples include a bicycle and a wheelbarrow.

**Clear v2:**

A compound machine combines two or more simple machines. Each part helps the whole machine work. A wheelbarrow is one example. Its handles help you lift a load, while its wheel helps you move it. A bicycle is another example.

## plain2k-everyday-train_sft-0575-qa4

**Frage:** That sounds interesting. What's a ryokan like?

**Vorher:**

A ryokan is a traditional Japanese inn that offers a unique cultural experience. They usually have tatami-matted rooms, futon beds, and shared baths. Many ryokans also offer traditional Japanese meals and activities, such as tea ceremonies.

**Clear v2:**

A ryokan is a traditional Japanese inn. Rooms often have woven floor mats called tatami. Guests may sleep on futons, which are bedding laid out on the floor. Some inns have shared baths and serve Japanese meals. Facilities and activities differ, so check the inn before booking.

## plain2k-everyday-train_sft-1098-qa2

**Frage:** I'm interested in learning about the moon phases. What are the main phases of the moon?

**Vorher:**

The main phases are new moon, waxing crescent, first quarter, waxing gibbous, full moon, waning gibbous, last quarter, and waning crescent. Waxing means the lit part we see is growing; waning means it is shrinking. A crescent is a thin curve, and gibbous means more than half looks lit.

**Clear v2:**

The cycle starts with a new moon, when the Moon looks dark. Then come waxing crescent, first quarter, and waxing gibbous. Next is the full moon. Then come waning gibbous, last quarter, and waning crescent. Waxing means the bright part we see is growing. Waning means it is shrinking. A crescent is a thin curve. Gibbous means more than half looks bright.

## plain2k-everyday-train_sft-1894-dialogue

**Frage:** I'm looking for information about different types of manufacturing jobs. Can you tell me about that?

**Vorher:**

There are several types of manufacturing jobs, including production line work, assembly, and quality control. These jobs often involve working with machinery, inspecting products, and following safety protocols.

**Clear v2:**

Manufacturing means making products. Workers may run machines or put parts together. Others check whether products are made correctly. Everyone needs to follow the safety rules for their job.
